import random
import re
import string
from typing import List, Pattern, Tuple, Optional

from query_translate.utils.type_defs import ChangeInfo


# RE pattern for searching slot variables (e.g ?!TERM!?) in template string
slot_search_pattern: Pattern = re.compile(r"\?![A-Z]+!\?")


class StringUtils:

    @staticmethod
    def apply_string_changes(original_s: str, fixes: List[ChangeInfo]) -> str:
        """
        Perform given fixes on the string
        Args:
            original_s: original string to be fixed
            fixes: fixes to be applied

        Returns:
            fixed string after application of the fixes
        """
        new_s = original_s

        # sort the fixes by index start
        s_list = sorted(fixes, key=lambda fx: fx.start_idx)

        # modify indexes by the string lengths
        len_incr = 0
        for fix in s_list:
            orig_start_idx = fix.start_idx
            fix.start_idx += len_incr
            new_s = new_s[:fix.start_idx] + fix.fixed_str + new_s[(fix.start_idx + len(fix.orig_str)):]
            len_incr += len(fix.fixed_str) - len(fix.orig_str)

            fix.start_idx = orig_start_idx
        return new_s

    @staticmethod
    def find_utterance_slots(slot_utterance: str) -> List[ChangeInfo]:
        """
        Find slots in a utterance template
        Args:
            slot_utterance: templated utterance/template, e.g. 'tables with ?!TERM!?'

        Returns:
            list of ChangeInfos with the slot information
        """
        changes = []
        for m in re.finditer(slot_search_pattern, slot_utterance):
            ch_info = ChangeInfo(slot_utterance[m.start():m.end()], "", m.start())
            changes.append(ch_info)
        return changes

    @staticmethod
    def trim_template_slot(template_slot: str) -> str:
        """ get entity name from slot, e.g. ?!TERM!? -> TERM """
        return template_slot[2:-2]

    @staticmethod
    def get_all_n_grams(temp_filtered_words: List[str]) -> List[Tuple[str]]:
        n_grams = []
        for i in range(1, len(temp_filtered_words)):
            sequences = [temp_filtered_words[j:] for j in range(i)]
            grams = list(zip(*sequences))
            n_grams.extend(grams)

        n_grams.append(tuple(temp_filtered_words))
        return n_grams

    @staticmethod
    def random_str(length: Optional[int] = None) -> str:
        """Generate random string of a given length."""
        length = length or random.randint(0, 1e3)
        chars = string.ascii_letters
        return ''.join(random.choices(chars, k=length))
