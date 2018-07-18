alphaumlautic = "[\u00C0-\u017Fa-zA-Z]"

endings_map = {'ENDINGS': '((e[rsmn]?)|e?s)?'}

speculation_regex_rules = u'''DD
könn
differential
eingeschränkte
beurteilbarkeit
suspekt
möglich
verdächtig
vermutl
verdacht
nicht[a-zöäüA-Z0-9[:blank:]]+ausgeschlossen
nicht[a-zöäüA-Z0-9[:blank:]]+ausschlie(ß|ss)bar'''
speculation_regex_rules_list = speculation_regex_rules.splitlines()
speculation_pattern = '^(' + '|'.join(speculation_regex_rules_list) + ')$'
speculation_map = {
    'SPECULATION': speculation_pattern
}

negation_map = {
    'NEGATION': '(kein((e[rsmn]?)|s)?|ohne)'
}

thorax_map = {
    'INFILTRAT': "\w*[iI]nfiltrat\w*",
    'PNEUMONIE': "\w*[pP]neumoni\w*",
    'DEKOKMPENSATION': "[dD]ekompens\w*",
    'KOMPENSATION': "[kK]ompens\w*",
    'STAUUNG': "\w*[sS]tauung\w*",
    'EMBOLIE': "\w*[eE]mbol\w*",
}


helper_map = {**negation_map, **endings_map}

icb_rules = u'''Intracraniell{ENDINGS} Blutung{ENDINGS}
H(ä|ae)morrhagi(e|sch{ENDINGS})?
(b|B)lutnachweis{ENDINGS}
Intrazerebral{ENDINGS} Blutung{ENDINGS}
{NEGATION}.* (\(?akut{ENDINGS})?.* Traumafolg{ENDINGS}
{NEGATION}.* traumatisch{ENDINGS}.* (Folg(en)?|Verletzung(en)?)'''.format_map(helper_map)

icb_map = {
    'ICB': '^(' + '|'.join(icb_rules.splitlines()) + ')$'
}

# (\W?\ws|S)phenoidfraktur
# (\W?\wf|F)elsenbeifraktur
# (\W?\ws|S)chädelbasisfraktur
# Einstrahl{ENDINGS} (d{ENDINGS})? (\wf|F)rakturspalt

# {NEGATION} (akuten )?([\u00C0-\u017Fa-zA-Z]*t|T)raumafolg
# {NEGATION} traumatisch{ENDINGS} ([\u00C0-\u017Fa-zA-Z]*f|F)olg|([\u00C0-\u017Fa-zA-Z]*v|V)erletzung)'''.format_map(helper_map)


fraktur_rules = u'''
(f|F)raktur{ENDINGS}
(f|F)issur{ENDINGS}
(l|L)ini{ENDINGS}
(b|B)ruch{ENDINGS}
(f|F)rakturspalt{ENDINGS}
'''.strip()

# no_negation_rule = '(?!(^|\W){NEGATION}\W)'.format_map(helper_map)
joined_rules = '[\u00C0-\u017Fa-zA-Z]*(' + '|'.join(fraktur_rules.splitlines()) + ')'

rule = joined_rules.format_map(helper_map) # + '{ENDINGS}($|\W)'
fraktur_map = {
    'FRAKTUR': rule
}


hydrocephalus_rules = u'''Liquorabflusstörung
keine Erweiterung des Ventrikelsystems
Ballonierte Ventrikel
Pellottierung der Vertrikel
normale Weite der Ventrikel'''
hydrocephalus = {
    'HYDROCEPHALUS': '^(' + '|'.join(hydrocephalus_rules.splitlines()) + ')$'
}

vessels_rules = '''regelrechte/normale/gute Kontrastierung der Gefässe
Arterien
keine Stenose
Verschluss
Dissektion
Elongation
elongierte Gefässe/Arterien
Blutgefässe'''

vessels_map = {
    'VESSELS': '^(' + '|'.join(vessels_rules.splitlines()) + ')$'
}


midline = '''Mittelständiger Interhemisphärenspalt
Mittellinie erhalten
mittelständiges Ventrikelsystem
Midline shift
Mittellinienverlagerung
Verlagerung der Mittellinie'''

negativ = '''Keine
Kein Nachweis
Keine Hinweise
keine Anhaltspunkte
normal(e)'''







