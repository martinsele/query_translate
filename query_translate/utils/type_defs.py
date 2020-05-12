from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict


Template = str
Utterance = str
Value = str


class DynTemplateFieldsEnum(Enum):
    """Definition of possible template types to be used in a query."""
    INTENT = "INTENT"       # intent such as SEARCH (with utterances such as 'list', 'find me', ...)
    SPECIAL = "SPECIAL"     # special templates (e.g. ?!SYSTEM!? admins, sensitive terms, ...)
    TYPE = "TYPE"           # entity type to search (e.g. catalogItem, attribute, rule, person, profile, ...)
    WITH = "WITH"           # define term on the entity
    FROM = "FROM"
    PERSON = "PERSON"


@dataclass(unsafe_hash=True)
class ChangeInfo:
    """Class containing information needed for string part replacement."""
    orig_str: str = field(compare=True)
    fixed_str: str = field(compare=False)
    start_idx: int = field(compare=True)

    def __repr__(self):
        return f"[ChangeInfo] {self.orig_str}({self.start_idx})"


# Dynamic template consists of dynamically assembled parts as [INTENT][TYPE][WITH][FROM][PERSON]
# e.g. {'INTENT': 'SEARCH', 'TYPE': 'catalogItem', 'WITH': 'with ?!TERM!?', 'FROM': 'from ?!SOURCE!?', 'PERSON': 'edited by ?!PERSON!?'}
# also the [TYPE] may be of a type SPECIAL in which case [WITH][FROM][PERSON] is not used
DynamicTemplate = Dict[DynTemplateFieldsEnum, Template]
DynamicValues = Dict[DynTemplateFieldsEnum, List[ChangeInfo]]
# TemplateTranslation is a translation of an utterance into list that maps each utterance word
# into template part where it belongs
# e.g. 'data from AWS' -> [TYPE, FROM, FROM]
TemplateTranslation = List[DynTemplateFieldsEnum]


@dataclass(frozen=True)
class DynamicDataSample:
    """Representation of a data sample - a query utterance, its template representation and required translation."""
    utterance: Utterance
    template: DynamicTemplate
    var_values: DynamicValues    # for each template field have a list of values
    translation: TemplateTranslation
