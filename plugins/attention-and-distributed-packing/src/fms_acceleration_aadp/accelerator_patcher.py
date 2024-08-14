
# singleton implementation of a dataloader patcher
# - DataLoaderPatcher.patch should only be called once

from accelerate import Accelerator

from enum import Enum
from dataclasses import dataclass

from transformers import DataCollator
from torch.utils.data import DataLoader

from typing import List, Dict, Any, Callable
from types import MethodType

# NOTE: this is more general than replacing components
class AcceleratorPatcherComponent(Enum):
    data_loader = 1
    data_collator = 2

# additional kwargs
RULE_SPECIAL_KWARGS = {
    AcceleratorPatcherComponent.data_loader.value: {"skip_prepare"}
}

# NOTE: these are those that can be replaced
REPLACEABLE_COMPONENTS_ENUMS = (
    AcceleratorPatcherComponent.data_loader,
    AcceleratorPatcherComponent.data_collator,
)

# DataCollator is a typing.NewType and does not work well with isinstance
# - so we replace it with Callable
REPLACEABLE_COMPONENTS_TYPES = {
    AcceleratorPatcherComponent.data_loader.value: DataLoader,
    AcceleratorPatcherComponent.data_collator.value: Callable
}

# helpful to keep a history of all patching that has been done
@dataclass
class AcceleratorPatcherHistory:

    # component that is patched
    component: AcceleratorPatcherComponent

    # name of the rule that was applied
    rule_id: str

@dataclass
class AcceleratorRuleReplace:

    # id, must be unique
    rule_id: str

    # component that is patched
    component: AcceleratorPatcherComponent

    # replacement:
    replacement: Any

    # pre-req check on the object to be replaced
    pre_req: Callable = None

    # additional kwargs that can be used for special behaviors based on component
    # e.g, skip_repare for dataloader will skip running the old prepare
    kwargs: Dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

        assert all(
            k in RULE_SPECIAL_KWARGS[self.component.value] for k in self.kwargs
        ), f"Invalid special behavior kwargs in '{self.kwargs.keys()}'"


    def pre_req_check(self, to_be_replaced: Any):
        if self.pre_req is None:
            return

        assert self.pre_req(to_be_replaced), \
            f"Rule '{self.rule_id}' failed pre-requisite check for type '{type(to_be_replaced)}'."

class AcceleratorPatcher:

    # singleton history of patches
    history: List[AcceleratorPatcherHistory] = []

    # singleton collection of replacements
    replacement_rules: Dict[str, AcceleratorRuleReplace] = {}

    @staticmethod
    def replace(
        rule_id: str, 
        component: AcceleratorPatcherComponent,
        replacement: Any, 
        pre_requisite_check : Callable = None,
        kwargs: Dict = None
    ):

        # generic check
        # ensure that rule has not been added before
        assert not any(
            h.rule_id == rule_id for h in AcceleratorPatcher.history
        ), f"Rule '{rule_id}' has already been added"

        # check for replacement rules
        # - if any of the replacements exist already
        if component in REPLACEABLE_COMPONENTS_ENUMS:

            assert isinstance(
                replacement, REPLACEABLE_COMPONENTS_TYPES[
                    component.value
                ]
            ), (
                f"Rule '{rule_id}' replacing component '{component}' with wrong ",
                f"type '{type(replacement)}'"
            )

        # store the replacment
        AcceleratorPatcher.replacement_rules[
            component.value
        ] = AcceleratorRuleReplace(
            rule_id, component,
            replacement, pre_requisite_check, kwargs
        )

        # record the history
        AcceleratorPatcher.history.append(
            AcceleratorPatcherHistory(component, rule_id)
        )

    @staticmethod
    def patch(accelerator: Accelerator):

        # some rules will require patching the prepare function
        if any(
            key in (
                AcceleratorPatcherComponent.data_collator.value,
                AcceleratorPatcherComponent.data_loader.value

            )
            for key in AcceleratorPatcher.replacement_rules
        ):
            AcceleratorPatcher._patch_prepare(accelerator)

    @staticmethod
    def _has_dataloader_replacement():
        return (
            AcceleratorPatcherComponent.data_loader in 
            AcceleratorPatcher.replacement_rules
        )

    # function to patch the accelerator prepare
    @staticmethod
    def _patch_prepare(accelerator: Accelerator):

        # hijack the dataloader in accelerator.prepare to replace the collate_fn
        _old_prepare = accelerator.prepare

        def prepare(self, *args, device_placement=None):
            if len(args) > 1 or not isinstance(args[0], DataLoader):
                return _old_prepare(*args, device_placement=device_placement)

            # if there is dataloader replacment
            replace_dataloader_rule = AcceleratorPatcher.replacement_rules.get(
                AcceleratorPatcherComponent.data_loader.value
            )
            # the original dataloader
            dataloader = args[0]

            if replace_dataloader_rule:
                replace_dataloader_rule.pre_req_check(dataloader)
                dataloader = replace_dataloader_rule.replacement

            # if there is dataloader replacment
            collator_replacement_rule = AcceleratorPatcher.replacement_rules.get(
                AcceleratorPatcherComponent.data_collator.value
            )

            if collator_replacement_rule:
                # - first we run the check on the rule (if any)
                # - then we replace
                collator_replacement_rule.pre_req_check(dataloader.collate_fn)
                # Replace the collate_fn in dataloader
                dataloader.collate_fn = collator_replacement_rule.replacement

            # - special behavior for dataloader replacements
            # - need to know if we run the original prepare
            if (
                replace_dataloader_rule is not None 
                and replace_dataloader_rule.kwargs.get("skip_prepare", False)
            ):
                return dataloader
            else:
                return _old_prepare(dataloader)

        accelerator.prepare = MethodType(prepare, accelerator)

    @staticmethod
    def summary():
        result = []
        result.append("***************** Accelerator Patching *************")
        for x in AcceleratorPatcher.history:
            result.append(
                "Rule: {0:15s} Component: {1:25s}".format(
                    x.rule_id, x.component
                )
            )

        return "\n".join(result)


def patch_accelerator(accelerator: Accelerator):
    AcceleratorPatcher.patch(accelerator)
