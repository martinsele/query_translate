from collections import OrderedDict
from typing import Dict, List, Tuple, Iterable

from query_translate.utils.type_defs import Template, Utterance, DynTemplateFieldsEnum


class DynamicGenValues:

    SOURCE_NAME = ["AWS", "Postgres", "MySQL", "DWS", "Local", "Cloud DB", "S3", "MariaDB", "S01"]

    TERM = ["customer", "sales", "PII", "GDPR", "currency", "Product ID", "SIN", "Gender", "E-mail", "email", "SSN",
            "credit card number", "phone", "person", 'sensitive', 'zip code', '[CAN] canada', 'address']

    SYSTEM = ["ONE", "DQD", "MDC", "RDM", "Catalog"]

    RULE = ["rule01", "xyz", "some rule1", "policy r02"]

    PERSON = ['me', 'myself', 'John Dow', 'Jane Dow', 'Peter Foo', 'Carl Bar', 'Robert Smith', 'Martin Tate',
              'Clara Reede', "Richard Mellow", 'Adele Whatever', 'Mathias Brown', "Julie Robertson", "Steve McQueen",
              "Arnold Black", "Tom Cruise"
              ]

    VAL_MAP = {"SOURCE": SOURCE_NAME,
               "TERM": TERM,
               "SYSTEM": SYSTEM,
               "RULE": RULE,
               "PERSON": PERSON}

    TEMPLATE_ORDER = [DynTemplateFieldsEnum.INTENT, DynTemplateFieldsEnum.SPECIAL, DynTemplateFieldsEnum.TYPE,
                      DynTemplateFieldsEnum.WITH, DynTemplateFieldsEnum.FROM, DynTemplateFieldsEnum.PERSON]

    def __init__(self):
        self.data = OrderedDict({
            DynTemplateFieldsEnum.INTENT: self.get_intent_utter(),
            DynTemplateFieldsEnum.SPECIAL: self.get_special_utter(),
            DynTemplateFieldsEnum.TYPE: self.get_type_utter(),
            DynTemplateFieldsEnum.WITH: self.get_with_utter(),
            DynTemplateFieldsEnum.FROM: self.get_from_utter(),
            DynTemplateFieldsEnum.PERSON: self.get_edited_by_utter()
        })

    @staticmethod
    def get_intent_utter() -> Dict[Template, List[Utterance]]:
        intent_templates = OrderedDict()
        intent_templates["SEARCH"] = ['show me', 'list all', 'show me all', '', 'search for', 'search', 'find all', 'find',
                                      'where are', 'what are the', 'what ar', 'get me', 'show']
        return intent_templates

    @staticmethod
    def get_type_utter() -> Dict[Template, List[Utterance]]:
        type_templates = OrderedDict()
        type_templates['catalogItem'] = ['catalogItem', 'catalog item', 'catalog-item', 'data', 'table',
                                         'catalogItems', 'catalog items', 'catalog-items', 'tables']
        type_templates['rule'] = ['rule', 'rules']
        type_templates['source'] = ['source', 'sources', 'database', 'databases', 'data-source', 'data-sources']
        type_templates['profile'] = ['profile', 'profiles', 'statistics']
        type_templates['attribute'] = ['attribute', 'attributes', 'column', 'columns']
        return type_templates

    @staticmethod
    def get_special_utter() -> Dict[Template, List[Utterance]]:
        templates = OrderedDict()
        template = "sensitive data"
        descriptions = ["sensitive data", "data with sensitive classification", "sensitive columns", "sensitive colmns",
                        "all sensitive data", "sensitiv data", "sensitive", "sensitive catalog items", "sensitive attributes"]
        templates[template] = descriptions

        template = "applications of rule ?!RULE!?"
        descriptions = ["tables where ?!RULE!? was applied", "all data covered by ?!RULE!? rule",
                        "where rule ?!RULE!? was applied", "applications of rule ?!RULE!?",
                        "aplications of the rule ?!RULE!?"]
        templates[template] = descriptions

        template = "all existing terms"
        descriptions = ["all existing terms", "existing terms", "all terms",
                        "terms that are defined", "terms in system", "all exsting terms"
                        ]
        templates[template] = descriptions

        template = "suggested terms"
        descriptions = ["suggested terms", "sugested terms", "terms that were suggested", "terms tht wer suggested",
                        "terms suggestions", "terms sugestions", "business terms suggestions", "suggested tags",
                        ]
        templates[template] = descriptions

        template = "sensitive terms"
        descriptions = ["sensitive terms", "all sensitive terms", "the sensitive terms",
                        "terms marked as sensitive", "terms markd like sensitive", "sensitiv terms",
                        "terms that are sensitive", "sensitve terms", "every sensitive term", "sensitive term"]
        templates[template] = descriptions

        template = "system users"
        descriptions = ["users", "system users", "all users", "existing persons in system",
                        "systm users", "persons using the system", "persons using the sstem"]
        templates[template] = descriptions

        template = "admins of ?!SYSTEM!?"
        descriptions = ["who has rights to ?!SYSTEM!?", "?!SYSTEM!? admins", "?!SYSTEM!? users",
                        "?!SYSTEM!? administrators", "?!SYSTEM!? owners"]
        templates[template] = descriptions

        # template = "users with admin rights"
        # descriptions = ["who has admin rights", "users with admin rights", "people with admin rights", "system admins",
        #                 "admins", "system administrators", "sstem admins", "administrators", "who are system admins"]
        # templates[template] = descriptions
        return templates

    @staticmethod
    def get_with_utter() -> Dict[Template, List[Utterance]]:
        templates = OrderedDict()
        template = "with ?!TERM!?"
        descriptions = ["with ?!TERM!?", "?!TERM!?", "wth ?!TERM!?", "labeled as ?!TERM!?", "having ?!TERM!? data in it",
                        "having ?!TERM!? tag", "with term ?!TERM!?", "using ?!TERM!?"]
        templates[template] = descriptions
        return templates

    @staticmethod
    def get_from_utter() -> Dict[Template, List[Utterance]]:
        templates = OrderedDict()
        template = "from ?!SOURCE!?"
        descriptions = ["from ?!SOURCE!?", "from ?!SOURCE!? source", "from ?!SOURCE!? database",
                        "from ?!SOURCE!? datalake", "located in ?!SOURCE!?"]
        templates[template] = descriptions
        return templates

    @staticmethod
    def get_edited_by_utter() -> Dict[Template, List[Utterance]]:
        templates = OrderedDict()
        template = "edited by ?!PERSON!?"
        descriptions = ["edited by ?!PERSON!?", "modified by ?!PERSON!?", "changed by ?!PERSON!?", "updated by ?!PERSON!?",
                        "?!PERSON!? updated", "?!PERSON!? changed", "?!PERSON!? modified", "?!PERSON!? edited"]
        templates[template] = descriptions
        return templates

    @staticmethod
    def encode_templates(templates: Iterable[Template]) -> Tuple[Dict[Template, int], Dict[int, Template]]:
        """
        Create mappings for template keys
        Args:
            templates: set of templates to encode

        Returns:
            regular (template -> idx) and inverse (idx -> template) mappings for templates
        """
        template_mapping = {}
        index_mapping = {}
        for idx, template in enumerate(templates):
            template_mapping[template] = idx
            index_mapping[idx] = template

        return template_mapping, index_mapping
