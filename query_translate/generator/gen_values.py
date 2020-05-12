"""Configurations and examples of query part templates and their values."""
from collections import OrderedDict
from typing import Dict, List, Tuple, Iterable

from query_translate.utils.type_defs import Template, Utterance

SOURCE_NAME = ["AWS", "Postgres", "MySQL", "DWS", "Local", "Cloud DB", "S3", "MariaDB", "S01"]

TERM = ["customer", "sales", "PII", "GDPR", "currency", "Product ID", "SIN", "Gender", "E-mail", "email", "SSN",
        "credit card number", "phone", "person"]

SYSTEM = ["ONE", "DQD", "MDC", "RDM", "Catalog"]

RULE = ["rule01", "xyz", "some rule", "policy r02"]

VAL_MAP = {"SOURCE": SOURCE_NAME,
           "TERM": TERM,
           "SYSTEM": SYSTEM,
           "RULE": RULE}


def generate_template_data(use_aql: bool = False) -> Dict[Template, List[Utterance]]:
    templates = OrderedDict()

    # region --------------- TABLES AND COLUMNS---------------
    aql_template = '$type in ("catalogItem") AND ' \
                   '(name like ??var1?? OR attributes.some(name like ??var1??) OR ' \
                   'termInstances.some(target{name like ??var1?? OR synonym like ??var1?? OR abbreviation like ??var1??}) OR ' \
                   'attributes.some(termInstances.some(target{name like ??var1?? OR synonym like ??var1?? OR abbreviation like ??var1??})))'
    template = "tables with ?!TERM!?"

    descriptions = ["tables with ?!TERM!?", "tbles with ?!TERM!?", "?!TERM!? data", "tables with ?!TERM!? data", "tables wth ?!TERM!? data",
                    "where are ?!TERM!? data", "everything labeled as ?!TERM!?", "find all ?!TERM!? entries", "tables with ?!TERM!? columns",
                    "tables having ?!TERM!? data in it", "catalog items having ?!TERM!? tag", "?!TERM!? catalog items",
                    "show me tables with term ?!TERM!?", "get me all ?!TERM!?", "show ?!TERM!? data"
                    ]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type in ("catalogItem", "attribute") AND termInstances.some(target{securityClassification="SENSITIVE"})'
    template = "sensitive data"

    descriptions = ["sensitive data", "data with sensitive classification", "sensitive columns", "sensitive colmns",
                    "all sensitive data", "where are sensitive data", "whereare sensitive data", "list everything sensitive",
                    "list sensitiv data", "wht is sensitive", "sensitive catalog items", "sensitive attributes"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # endregion --------------- TABLES ---------------

    # region --------------- TABLES AND COLUMNS WITH SOURCES ---------------
    aql_template = '$type in ("catalogItem") AND ' \
                   '(name like ??var1?? OR attributes.some(name like ??var1??) OR ' \
                   'termInstances.some(target{name like ??var1?? OR synonym like ??var1?? OR abbreviation like ??var1??}) OR ' \
                   'attributes.some(termInstances.some(target{name like ??var1?? OR synonym like ??var1?? OR abbreviation like ??var1??}))) AND ' \
                   '$parent.$parent.name like ??var2??'
    template = "tables with ?!TERM!? from ?!SOURCE!?"

    descriptions = ["tables with ?!TERM!? from ?!SOURCE!?", "?!TERM!? data from ?!SOURCE!? source", "catalog items with ?!TERM!? from ?!SOURCE!? database"
                    "tables with ?!TERM!? data from ?!SOURCE!? datalake", "catalog items having ?!TERM!? tag from ?!SOURCE!?", "get me all ?!TERM!? from ?!SOURCE!?",
                    "list ?!TERM!? located in ?!SOURCE!?", "show ?!TERM!? data from ?!SOURCE!?"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type in ("catalogItem", "attribute") AND ' \
                   'termInstances.some(target{securityClassification="SENSITIVE"}) AND ' \
                   '($parent.$parent.name like ??var1?? OR $parent.$parent.$parent.name like ??var1??'
    template = "sensitive data from ?!SOURCE!?"

    descriptions = ["sensitive data from ?!SOURCE!?", "data with sensitive classification from ?!SOURCE!? source",
                    "sensitive columns from ?!SOURCE!? data", "sensitive entries in ?!SOURCE!?", "sensitive items in ?!SOURCE!?",
                    "sensitive assets in ?!SOURCE!?", "show me sensitive data from ?!SOURCE!?"
                    ]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type in ("catalogItem") AND $parent.$parent.name like ??var1??'
    template = "data from ?!SOURCE!?"
    descriptions = ["data from ?!SOURCE!? database", "data from source ?!SOURCE!?", "all data from ?!SOURCE!? system",
                    "all tables in ?!SOURCE!?", "list every catalog item in ?!SOURCE!? system", "catalog items from ?!SOURCE!?"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # endregion --------------- TABLES ---------------

    # region --------------- RULES ---------------
    aql_template = '$type in ("rule") AND ' \
                   'inputGroups.some(inputs.some(termInstances.some(target{name like ??var1??}))) OR ' \
                   '$type in ("ruleInstance") AND ' \
                   'target{inputGroups.some(inputs.some(termInstances.some(target{name like ??var1??})))} OR ' \
                   '$type in ("metadataRule") AND ' \
                   'termInstances.some(target{name like ??var1??})'
    template = "rules using ?!TERM!?"

    descriptions = ["rules with ?!TERM!?", "rules using ?!TERM!?", "rules having ?!TERM!?", "show me rules with ?!TERM!?",
                    "list rules using ?!TERM!? tag"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = 'termInstances.some(target {detectionRules.ruleInstances.some(target {name like ??var1?? OR synonym like ??var1?? OR abbreviation like ??var1??})})'
    template = "applications of rule ?!RULE!?"

    descriptions = ["tables where ?!RULE!? was applied", "all data covered by ?!RULE!? rule",
                    "find instances where rule ?!RULE!? was applied", "applications of rule ?!RULE!?", "aplications of the rule ?!RULE!?"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type in ("term", "businessTerm", "technicalTerm")'
    template = "all existing terms"
    descriptions = ["show all existing terms", "all existing terms", "list all existing terms", "list all terms", "what terms are defined",
                    "lst all terms", "what terms ar in system", "all exsting terms"
                    ]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # endregion --------------- RULES ---------------

    # region --------------- TERMS  ---------------
    # aql_template = '$type in ("term", "businessTerm", "technicalTerm") AND ' \
    #                '(name like ??var1?? OR synonym like ??var1?? OR abbreviation like ??var1??)'
    # template = "terms with ?!TERM!? tag"
    # descriptions = ["?!TERM!? terms", "terms with ?!TERM!? tag", "?!TERM!? tags", "tags like ?!TERM!?", "show me ?!TERM!? terms"]
    # if use_aql:
    #     templates[aql_template] = descriptions
    # else:
    #     templates[template] = descriptions

    # ---------------
    aql_template = '$type="termSuggestion"'
    template = "suggested terms"
    descriptions = ["suggested terms", "sugested terms", "terms that were suggested", "terms tht wer suggested",
                    "show me suggested terms", "shw me sugested terms", "what terms were suggested",
                    "terms suggestions", "terms sugestions", "business terms suggestions", "list all suggested tags",
                    ]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type="businessTerm" AND securityClassification="SENSITIVE"'
    template = "sensitive terms"
    descriptions = ["sensitive terms", "all sensitive terms", "what are the sensitive terms", "what are sensitive terms",
                    "list terms marked as sensitive", "list terms markd like sensitive", 'sensitiv terms'
                    "lst terms that are sensitive", "list sensitve terms", "every sensitive term", "sensitive term"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # endregion --------------- TERMS  ---------------

    # region --------------- USERS  ---------------
    aql_template = '$type="person"'
    template = "system users"
    descriptions = ["what are the users", "system users", "show me all users", "existing persons in system",
                    "list all users", "all users", "systm users", "persons using the system", "persons using the sstem"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type="person" AND roles.some(role{name like ??var1?? or description like ??var1??})'
    template = "?!SYSTEM!? admins"
    descriptions = ["who has rights to ?!SYSTEM!?", "?!SYSTEM!? admins", "?!SYSTEM!? users", "?!SYSTEM!? administrators", "?!SYSTEM!? owners"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # ---------------
    aql_template = '$type="person" AND roles.some(role{name like "admin" or description like "admin"})'
    template = "users with admin rights"
    descriptions = ["who has admin rights", "users with admin rights", "people with admin rights", "system admins", "admins",
                    "system administrators", "sstem admins", "administrators", "who are system admins"]
    if use_aql:
        templates[aql_template] = descriptions
    else:
        templates[template] = descriptions

    # endregion --------------- USERS  ---------------
    return templates


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
