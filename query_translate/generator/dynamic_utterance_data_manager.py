import itertools
import random
from typing import Dict, Tuple, List, Optional, Iterator

import numpy as np

from query_translate.generator.dynamic_gen_values import DynamicGenValues
from query_translate.utils.string_utils import StringUtils
from query_translate.utils.type_defs import (Template, Utterance, DynamicDataSample, DynamicTemplate,
                                             DynTemplateFieldsEnum, DynamicValues, TemplateTranslation, ChangeInfo)


class DynamicUtteranceDataManager:
    """
    Class for generating and managing data samples using dynamic templates
    """
    data_utils: DynamicGenValues     # prepared prescriptions for data generation
    dataset: List[DynamicDataSample]
    type_template_map: Dict[Template, int]  # templates mapping for entity types
    type_idx_map: Dict[int, Template]

    special_template_map: Dict[Template, int]   # templates mapping for special utterances
    special_idx_map: Dict[int, Template]

    def __init__(self, random_state: int = 1, generate_data: bool = False):
        random.seed(random_state)
        self.data_utils = DynamicGenValues()
        if generate_data:
            self.dataset = self._generate_data(random_state)
        else:
            self.dataset = []

    def get_utterances(self) -> List[DynamicDataSample]:
        """ Return list of generated data samples"""
        return self.dataset

    def encode_template(self, template: DynamicTemplate) -> np.ndarray:
        """
        Encode dynamic template to array as [INTENT, SPECIAL, TYPE, WITH, FROM, PERSON], where
         INTENT ~ [1 = SEARCH]
         SPECIAL ~ [0 = not special, 1 = sensitive data, 2 = applications of rule ?!RULE!?, ...]
         TYPE ~ [0 = no type, 1 = catalogItem, 2 = rule, 3 = source, ...]
         WITH ~ [0 = no with part, 1 = has with part]
         FROM ~ [0 = no from part, 1 = has from part]
         PERSON ~ [0 = no edited by part, 1 = has edited by part]
        """
        y_array = np.zeros([1, 6])
        for idx, template_part_type in enumerate(DynamicGenValues.TEMPLATE_ORDER):
            y_array[0, idx] = self.get_template_part_encoding(template, template_part_type)
        return y_array

    def get_template_part_encoding(self, template: DynamicTemplate, template_part_type: DynTemplateFieldsEnum) -> int:
        """ Return template encoding for given template part type """
        part_value = template.get(template_part_type, None)
        if part_value is None:
            part_encode = 0
        else:
            part_encode = list(self.data_utils.data[template_part_type].keys()).index(part_value) + 1
        return part_encode

    def decode_template(self, template_array: np.ndarray) -> DynamicTemplate:
        """ Decode Dynamic template from numpy array """
        dyn_template = {}
        for template_part_type in DynamicGenValues.TEMPLATE_ORDER:
            decoded = self.decode_template_part(template_array, template_part_type)
            if decoded is not None:
                dyn_template[template_part_type] = decoded
        return dyn_template

    def decode_template_part(self, template_array: np.ndarray, template_part_type: DynTemplateFieldsEnum) -> Optional[Template]:
        """ Decode one particular part of a dynamic template """
        idx = DynamicGenValues.TEMPLATE_ORDER.index(template_part_type)
        y_val = int(template_array[idx])
        return self.decode_template_part_from_value(y_val, template_part_type)

    def decode_template_part_from_value(self, template_val: int, template_part_type: DynTemplateFieldsEnum) -> Optional[Template]:
        """ decode one particular part of a dynamic template given the value actual template value """
        if template_val == 0:
            return None
        return list(self.data_utils.data[template_part_type].keys())[template_val - 1]

    def _generate_data(self, random_state: int = 1) -> List[DynamicDataSample]:
        """
        Randomly generate utterances by the following procedure:
            1. Prepare all possible templates for special and regular template cases
            2. Generate utterances for each template:
                a. get all utterances for each of the templates part
                b. make product of all possibilities
                c. go over each product element, randomly select entity to fill it
        Args:
            random_state: state to use for random template values fillings
        Returns:
            List of DynamicDataSamples
        """
        random.seed(random_state)

        # Prepare templates
        templ_data = []
        templ_data.extend(self._gen_special_case_templates())    # special templates
        templ_data.extend(self._gen_non_special_templates())     # build other templates
        # Generate utterances
        empty_utterances = self._get_empty_utterances(templ_data)
        filled_utterances = self._fill_slots(empty_utterances)
        return filled_utterances

    @classmethod
    def generate_data_random(cls) -> Iterator[DynamicDataSample]:
        """
        Randomly generate utterances by the following procedure:
        Returns:
            generator of DynamicDataSamples
        """
        data_prep = DynamicGenValues()

        while True:
            # get intent:
            template = {DynTemplateFieldsEnum.INTENT: "SEARCH"}
            translation = []
            utter = random.choice(data_prep.data[DynTemplateFieldsEnum.INTENT]["SEARCH"])
            if utter != "":
                translation = [DynTemplateFieldsEnum.INTENT] * len(utter.split())

            # special or type
            has_special = random.random() < 0.2
            if has_special:
                utter_s = cls._add_single_type(data_prep.data, DynTemplateFieldsEnum.SPECIAL, template, translation)
                utter = " ".join([utter, utter_s])
            else:
                # type
                utter_s = cls._add_single_type(data_prep.data, DynTemplateFieldsEnum.TYPE, template, translation)
                utter = " ".join([utter, utter_s])
                # other template parts
                other_templ_parts = data_prep.TEMPLATE_ORDER[3:].copy()
                random.shuffle(other_templ_parts)
                for t_part in other_templ_parts:
                    has_part = random.random() < 0.6
                    if has_part:
                        utter_s = cls._add_single_type(data_prep.data, t_part, template, translation)
                        utter = " ".join([utter, utter_s])

            filled = cls._fill_slots_random(utter, template, translation, mode="random_str")  # 'random_choice'
            dyn_vals, utter = filled
            filled_data = DynamicDataSample(utterance=utter, template=template,
                                            var_values=dyn_vals, translation=translation)

            yield filled_data

    @staticmethod
    def _add_single_type(data: Dict, template_part_type: DynTemplateFieldsEnum,
                         template: DynamicTemplate, translation: TemplateTranslation) -> Utterance:
        type_templ = random.choice(list(data[template_part_type].keys()))
        utter = random.choice(data[template_part_type][type_templ])
        template[template_part_type] = type_templ
        translation.extend([template_part_type] * len(utter.split()))
        return utter

    @staticmethod
    def _gen_special_case_templates() -> List[DynamicTemplate]:
        """ Generate special cases templates """
        templ_data = []
        special_utter = DynamicGenValues.get_special_utter()
        for spec_templ, utter_list in special_utter.items():
            template = {DynTemplateFieldsEnum.INTENT: "SEARCH", DynTemplateFieldsEnum.SPECIAL: spec_templ}
            templ_data.append(template)
        return templ_data

    def _gen_non_special_templates(self) -> List[DynamicTemplate]:
        """ Generate dynamic templates using different template parts """
        type_templ = self._gen_type_templates()
        type_templ = self._append_template_part(type_templ, DynTemplateFieldsEnum.WITH)
        type_templ = self._append_template_part(type_templ, DynTemplateFieldsEnum.FROM)
        type_templ = self._append_template_part(type_templ, DynTemplateFieldsEnum.PERSON)
        return type_templ

    @staticmethod
    def _gen_type_templates() -> List[DynamicTemplate]:
        """ Generate templates for types """
        templ_data = []
        type_utter = DynamicGenValues.get_type_utter()
        for type_templ, utter_list in type_utter.items():
            template = {DynTemplateFieldsEnum.INTENT: "SEARCH", DynTemplateFieldsEnum.TYPE: type_templ}
            templ_data.append(template)
        return templ_data

    def _append_template_part(self, concat_templs: List[DynamicTemplate], part_type: DynTemplateFieldsEnum) \
            -> List[DynamicTemplate]:
        templ_data = []
        part_templates = list(self.data_utils.data[part_type].keys())
        part_templates.append(None)  # for case that does not use this template part
        combs = itertools.product(concat_templs, part_templates)

        for c_tmpl, p_templ in combs:
            new_templ = c_tmpl.copy()
            if p_templ is not None:
                new_templ[part_type] = p_templ
            templ_data.append(new_templ)
        return templ_data

    def _get_empty_utterances(self, templ_data: List[DynamicTemplate]) \
            -> List[Tuple[DynamicTemplate, List[Utterance], List[TemplateTranslation]]]:
        """
        Get utterances without filled slots from the templates
        Args:
            templ_data: list of generated dynamic templates
        Returns:
            utterances without filled slots from the templates mapped by the templates
        """
        empty_utterances = []
        # go over all pre-generated dynamic templates
        for dyn_templ in templ_data:
            utterances = [""]     # list of currently build utterances mapped by the template
            translations = [[]]

            # go by template field types by predefined order skipping [INTENT], i.e. [TYPE] [WITH] ...
            for templ_field_type in self.data_utils.TEMPLATE_ORDER[1:]:
                templ_field = dyn_templ.get(templ_field_type, None)
                # if template field present, add utterances
                if templ_field is not None:
                    utterances, translations = self._add_utterances_translations_combs(templ_field_type, templ_field,
                                                                                       utterances, translations)
            # randomly add intents
            intent_type = dyn_templ[DynTemplateFieldsEnum.INTENT]
            random_intents = random.choices(self.data_utils.data[DynTemplateFieldsEnum.INTENT][intent_type],
                                            k=len(utterances))
            intent_transl = self.translate_template_parts(random_intents, DynTemplateFieldsEnum.INTENT)
            utterances = [" ".join([ri, u]).strip() for ri, u in zip(random_intents, utterances)]
            translations = [list(itertools.chain(it, t)) for it, t in zip(intent_transl, translations)]
            empty_utterances.append((dyn_templ, utterances, translations))
        return empty_utterances

    def _add_utterances_translations_combs(self, templ_field_type: DynTemplateFieldsEnum, template_field: str,
                                           utterances: List[Utterance], translations: List[TemplateTranslation]) \
            -> Tuple[List[Utterance], List[TemplateTranslation]]:
        """
        Add combinations of utterances and their translations into the corresponding lists
        Args:
            templ_field_type: type of template part utterances to be added
            template_field: which one of the particular template parts of given type to be added
            utterances: current list of utterances to be extended
            translations: current list of utterance translations to be extended
        Returns:
            extended lists of utterances and translations
        """
        # create and extend list of utterances for template part and value ~ ['using ?!TERM!?', 'with ?!TERM!?', ..]}
        t_utter_vals = self.data_utils.data[templ_field_type][template_field]
        utter_combs = itertools.product(utterances, t_utter_vals)
        utterances = [" ".join(utter_part).strip() for utter_part in utter_combs]
        # create and extend list of template translations
        part_transls = self.translate_template_parts(t_utter_vals, templ_field_type)
        transl_combs = itertools.product(translations, part_transls)
        translations = [list(itertools.chain.from_iterable(transl)) for transl in transl_combs]
        return utterances, translations

    def _fill_slots(self, empty_utters: List[Tuple[DynamicTemplate, List[Utterance], List[TemplateTranslation]]]) \
            -> List[DynamicDataSample]:
        """  Fill empty utterances """
        filled_data = []
        for templ, utter_list, transl_list in empty_utters:
            for e_utter, e_transl in zip(utter_list, transl_list):
                filled = self._fill_slots_random(e_utter, templ, e_transl, mode="random_choice")   # "random_str" fill the empty slots with random values
                dyn_vals, utter = filled
                filled_data.append(DynamicDataSample(utterance=utter, template=templ,
                                                     var_values=dyn_vals, translation=e_transl))
        return filled_data

    @staticmethod
    def translate_template_parts(templ_utters: List[Utterance], templ_part_type: DynTemplateFieldsEnum) \
            -> List[TemplateTranslation]:
        """ Translate the utterance of a template part into a TemplateTranslation """
        translations = []
        for utter in templ_utters:
            words = utter.split()
            transl = [templ_part_type] * len(words)
            translations.append(transl)
        return translations

    @classmethod
    def _fill_slots_random(cls, slotted_utterance: Utterance, template: DynamicTemplate,
                           translation: TemplateTranslation, mode='random_choice') -> Tuple[DynamicValues, Utterance]:
        """
        Fill utterance with empty slots with random values by their type
        Args:
            slotted_utterance: e.g. 'show me data with ?!TERM!? tag from ?!SOURCE!?'
            template: template to follow while creating DynamicValues,
                      e.g. {'INTENT':'SEARCH', 'TYPE':'catalogItem', 'WITH': 'with ?!TERM!?, 'FROM': 'from ?!SOURCE!?'}
            translation: template translation to modify, if more than 1 word filled into slots
            mode: how to generate slot values
        Returns:
            filled utterance, e.g. 'show me data with customer tag from AWS'
        """
        slots = StringUtils.find_utterance_slots(slotted_utterance)
        word_list = slotted_utterance.split()
        dyn_values = {}
        for templ_k, templ_v in template.items():
            templ_slots = StringUtils.find_utterance_slots(templ_v)
            if len(templ_slots) > 0:
                values = []
                for t_slot in templ_slots:
                    val_to_fill = cls._select_fill_value(t_slot, mode=mode)
                    # find template slot in the utterance slots
                    found_slots = list(filter(lambda x: x.orig_str == t_slot.orig_str and not x.fixed_str, slots))

                    found_slots[0].fixed_str = val_to_fill
                    values.append(found_slots[0])

                    num_added_words = len(val_to_fill.split())
                    if num_added_words > 1:
                        idx = word_list.index(t_slot.orig_str)
                        list_update = [templ_k] * (num_added_words-1)
                        translation[idx:idx] = list_update
                        word_list[idx:idx] = list_update    # add also into original word list to keep correct indexing
                dyn_values[templ_k] = values

        filled_utter = StringUtils.apply_string_changes(slotted_utterance, slots)
        return dyn_values, filled_utter

    @staticmethod
    def _select_fill_value(slot_to_fill: ChangeInfo, mode='random_choice'):
        """ Prepare value to fill into a template slot """
        val_to_fill = 'xxx xxx'
        if mode == 'random_choice':
            possible_vals = DynamicGenValues.VAL_MAP[StringUtils.trim_template_slot(slot_to_fill.orig_str)]
            val_to_fill = random.choice(possible_vals)
        elif mode == 'random_str':
            val_to_fill = StringUtils.random_str(4)
        return val_to_fill

    @staticmethod
    def train_test_utter_split(data_samples: List[DynamicDataSample], test_ratio: float = 0.3, random_state: int = 1) \
            -> Tuple[List[DynamicDataSample], List[DynamicDataSample]]:
        """
        Split X a y arrays into train and test data according to given ratio
        Args:
            data_samples: list of data samples
            test_ratio: ratio of test data to N
            random_state: the seed used by the random number generator

        Returns:
            pair like train_data, test_data
        """
        num_utter = len(data_samples)
        tst_idx = int(num_utter * test_ratio)
        random.seed(random_state)
        shuffled_ids = list(range(num_utter))
        random.shuffle(shuffled_ids)

        test_data = [data_samples[idx] for idx in shuffled_ids[:tst_idx]]
        train_data = [data_samples[idx] for idx in shuffled_ids[tst_idx:]]
        return train_data, test_data

    @staticmethod
    def template_translation_to_text(translation: TemplateTranslation) -> str:
        return " ".join(t.value for t in translation)

    @staticmethod
    def text_to_template_translation(text: str) -> TemplateTranslation:
        translation = []
        for word in text.split():
            translation.append(DynTemplateFieldsEnum(word))
        return translation

    def save_dataset(self):
        with open("data.txt", mode='wt') as file:
            gen = (f"{sample.utterance} -- {self.template_translation_to_text(sample.translation)}\n" for
                   sample in self.dataset)
            file.writelines(gen)


if __name__ == "__main__":
    random.seed(1)
    data_gen = DynamicUtteranceDataManager.generate_data_random()
    for i, s in enumerate(data_gen):
        print(s.utterance)
        print(DynamicUtteranceDataManager.template_translation_to_text(s.translation))
        print("----------")
        if i > 10:
            break
