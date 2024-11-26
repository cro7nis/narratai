import re

class Preprocessor:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def split_sentence_to_max_tokens(text, tokenizer, special_token='▁', max_tokens=210, unk_token='<unk>'):
        tokens = tokenizer.tokenize(text)
        # print(tokens)

        batches = [re.sub("\s\s+", " ", ''.join(sentence).replace(special_token, ' ').replace(unk_token, ' ').strip())
                   for sentence in Preprocessor.split(tokens, max_tokens)]
        # print(len(tokens), len(batches), batches)
        return batches

    @staticmethod
    def split(list_a, chunk_size):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]

    @staticmethod
    def preprocess_input(text, tokenizer, special_token='▁', max_tokens=210, unk_token='<unk>',
                         split_on_punctuation=True):
        text = re.sub(r'((?<![A-Z])\.(?!\s))', r'\1 ', text)
        if split_on_punctuation:
            texts = re.split('(?<=[.!?;]) +', text)  # split on .!?;
        else:
            texts = [text]

        texts_list = []
        for text in texts:
            texts_list.extend(
                Preprocessor.split_sentence_to_max_tokens(text, tokenizer, max_tokens=max_tokens, unk_token=unk_token,
                                                          special_token=special_token))
        return texts_list

    def __call__(self, text, tokenizer, special_token='▁', max_tokens=210, unk_token='<unk>',
                 split_on_punctuation=True):
        return Preprocessor.preprocess_input(text, tokenizer, special_token=special_token, max_tokens=max_tokens,
                                             unk_token=unk_token, split_on_punctuation=split_on_punctuation)

iso_codes_str = '''aa	Afar
ab	Abkhazian
ae	Avestan
af	Afrikaans
ak	Akan
am	Amharic
an	Aragonese
ar	Arabic
as	Assamese
av	Avaric
ay	Aymara
az	Azerbaijani
ba	Bashkir
be	Belarusian
bg	Bulgarian
bh	Bihari languages
bm	Bambara
bi	Bislama
bn	Bengali
bo	Tibetan
br	Breton
bs	Bosnian
ca	Catalan; Valencian
ce	Chechen
ch	Chamorro
co	Corsican
cr	Cree
cs	Czech
cu	Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic
cv	Chuvash
cy	Welsh
da	Danish
de	German
dv	Divehi; Dhivehi; Maldivian
dz	Dzongkha
ee	Ewe
el	Greek, Modern (1453-)
en	English
eo	Esperanto
es	Spanish; Castilian
et	Estonian
eu	Basque
fa	Persian
ff	Fulah
fi	Finnish
fj	Fijian
fo	Faroese
fr	French
fy	Western Frisian
ga	Irish
gd	Gaelic; Scottish Gaelic
gl	Galician
gn	Guarani
gu	Gujarati
gv	Manx
ha	Hausa
he	Hebrew
hi	Hindi
ho	Hiri Motu
hr	Croatian
ht	Haitian; Haitian Creole
hu	Hungarian
hy	Armenian
hz	Herero
ia	Interlingua (International Auxiliary Language Association)
id	Indonesian
ie	Interlingue; Occidental
ig	Igbo
ii	Sichuan Yi; Nuosu
ik	Inupiaq
io	Ido
is	Icelandic
it	Italian
iu	Inuktitut
ja	Japanese
jv	Javanese
ka	Georgian
kg	Kongo
ki	Kikuyu; Gikuyu
kj	Kuanyama; Kwanyama
kk	Kazakh
kl	Kalaallisut; Greenlandic
km	Central Khmer
kn	Kannada
ko	Korean
kr	Kanuri
ks	Kashmiri
ku	Kurdish
kv	Komi
kw	Cornish
ky	Kirghiz; Kyrgyz
la	Latin
lb	Luxembourgish; Letzeburgesch
lg	Ganda
li	Limburgan; Limburger; Limburgish
ln	Lingala
lo	Lao
lt	Lithuanian
lu	Luba-Katanga
lv	Latvian
mg	Malagasy
mh	Marshallese
mi	Maori
mk	Macedonian
ml	Malayalam
mn	Mongolian
mr	Marathi
ms	Malay
mt	Maltese
my	Burmese
na	Nauru
nb	Bokmål, Norwegian; Norwegian Bokmål
nd	Ndebele, North; North Ndebele
ne	Nepali
ng	Ndonga
nl	Dutch; Flemish
nn	Norwegian Nynorsk; Nynorsk, Norwegian
no	Norwegian
nr	Ndebele, South; South Ndebele
nv	Navajo; Navaho
ny	Chichewa; Chewa; Nyanja
oc	Occitan (post 1500)
oj	Ojibwa
om	Oromo
or	Oriya
os	Ossetian; Ossetic
pa	Panjabi; Punjabi
pi	Pali
pl	Polish
ps	Pushto; Pashto
pt	Portuguese
qu	Quechua
rm	Romansh
rn	Rundi
ro	Romanian; Moldavian; Moldovan
ru	Russian
rw	Kinyarwanda
sa	Sanskrit
sc	Sardinian
sd	Sindhi
se	Northern Sami
sg	Sango
si	Sinhala; Sinhalese
sk	Slovak
sl	Slovenian
sm	Samoan
sn	Shona
so	Somali
sq	Albanian
sr	Serbian
ss	Swati
st	Sotho, Southern
su	Sundanese
sv	Swedish
sw	Swahili
ta	Tamil
te	Telugu
tg	Tajik
th	Thai
ti	Tigrinya
tk	Turkmen
tl	Tagalog
tn	Tswana
to	Tonga (Tonga Islands)
tr	Turkish
ts	Tsonga
tt	Tatar
tw	Twi
ty	Tahitian
ug	Uighur; Uyghur
uk	Ukrainian
ur	Urdu
uz	Uzbek
ve	Venda
vi	Vietnamese
vo	Volapük
wa	Walloon
wo	Wolof
xh	Xhosa
yi	Yiddish
yo	Yoruba
za	Zhuang; Chuang
zh	Chinese
zu	Zulu'''
iso_codes_str = iso_codes_str.split('\n')
lang2iso = {}
for code in iso_codes_str:
    iso, lang = code.split('\t')
    lang2iso[re.split(', |;| ', lang)[0]] = iso

codes_as_string = '''Acehnese (Arabic script)	ace_Arab
Acehnese (Latin script)	ace_Latn
Mesopotamian Arabic	acm_Arab
Ta’izzi-Adeni Arabic	acq_Arab
Tunisian Arabic	aeb_Arab
Afrikaans	afr_Latn
South Levantine Arabic	ajp_Arab
Akan	aka_Latn
Amharic	amh_Ethi
North Levantine Arabic	apc_Arab
Arabic	arb_Arab
Modern Standard Arabic (Romanized)	arb_Latn
Najdi Arabic	ars_Arab
Moroccan Arabic	ary_Arab
Egyptian Arabic	arz_Arab
Assamese	asm_Beng
Asturian	ast_Latn
Awadhi	awa_Deva
Central Aymara	ayr_Latn
South Azerbaijani	azb_Arab
North Azerbaijani	azj_Latn
Bashkir	bak_Cyrl
Bambara	bam_Latn
Balinese	ban_Latn
Belarusian	bel_Cyrl
Bemba	bem_Latn
Bengali	ben_Beng
Bhojpuri	bho_Deva
Banjar (Arabic script)	bjn_Arab
Banjar (Latin script)	bjn_Latn
Standard Tibetan	bod_Tibt
Bosnian	bos_Latn
Buginese	bug_Latn
Bulgarian	bul_Cyrl
Catalan	cat_Latn
Cebuano	ceb_Latn
Czech	ces_Latn
Chokwe	cjk_Latn
Central Kurdish	ckb_Arab
Crimean Tatar	crh_Latn
Welsh	cym_Latn
Danish	dan_Latn
German	deu_Latn
Southwestern Dinka	dik_Latn
Dyula	dyu_Latn
Dzongkha	dzo_Tibt
Greek	ell_Grek
English	eng_Latn
Esperanto	epo_Latn
Estonian	est_Latn
Basque	eus_Latn
Ewe	ewe_Latn
Faroese	fao_Latn
Fijian	fij_Latn
Finnish	fin_Latn
Fon	fon_Latn
French	fra_Latn
Friulian	fur_Latn
Nigerian Fulfulde	fuv_Latn
Scottish Gaelic	gla_Latn
Irish	gle_Latn
Galician	glg_Latn
Guarani	grn_Latn
Gujarati	guj_Gujr
Haitian Creole	hat_Latn
Hausa	hau_Latn
Hebrew	heb_Hebr
Hindi	hin_Deva
Chhattisgarhi	hne_Deva
Croatian	hrv_Latn
Hungarian	hun_Latn
Armenian	hye_Armn
Igbo	ibo_Latn
Ilocano	ilo_Latn
Indonesian	ind_Latn
Icelandic	isl_Latn
Italian	ita_Latn
Javanese	jav_Latn
Japanese	jpn_Jpan
Kabyle	kab_Latn
Jingpho	kac_Latn
Kamba	kam_Latn
Kannada	kan_Knda
Kashmiri (Arabic script)	kas_Arab
Kashmiri (Devanagari script)	kas_Deva
Georgian	kat_Geor
Central Kanuri (Arabic script)	knc_Arab
Central Kanuri (Latin script)	knc_Latn
Kazakh	kaz_Cyrl
Kabiyè	kbp_Latn
Kabuverdianu	kea_Latn
Khmer	khm_Khmr
Kikuyu	kik_Latn
Kinyarwanda	kin_Latn
Kyrgyz	kir_Cyrl
Kimbundu	kmb_Latn
Northern Kurdish	kmr_Latn
Kikongo	kon_Latn
Korean	kor_Hang
Lao	lao_Laoo
Ligurian	lij_Latn
Limburgish	lim_Latn
Lingala	lin_Latn
Lithuanian	lit_Latn
Lombard	lmo_Latn
Latgalian	ltg_Latn
Luxembourgish	ltz_Latn
Luba-Kasai	lua_Latn
Ganda	lug_Latn
Luo	luo_Latn
Mizo	lus_Latn
Standard Latvian	lvs_Latn
Magahi	mag_Deva
Maithili	mai_Deva
Malayalam	mal_Mlym
Marathi	mar_Deva
Minangkabau (Arabic script)	min_Arab
Minangkabau (Latin script)	min_Latn
Macedonian	mkd_Cyrl
Plateau Malagasy	plt_Latn
Maltese	mlt_Latn
Meitei (Bengali script)	mni_Beng
Halh Mongolian	khk_Cyrl
Mossi	mos_Latn
Maori	mri_Latn
Burmese	mya_Mymr
Dutch	nld_Latn
Norwegian Nynorsk	nno_Latn
Norwegian Bokmål	nob_Latn
Nepali	npi_Deva
Northern Sotho	nso_Latn
Nuer	nus_Latn
Nyanja	nya_Latn
Occitan	oci_Latn
West Central Oromo	gaz_Latn
Odia	ory_Orya
Pangasinan	pag_Latn
Eastern Panjabi	pan_Guru
Papiamento	pap_Latn
Western Persian	pes_Arab
Polish	pol_Latn
Portuguese	por_Latn
Dari	prs_Arab
Southern Pashto	pbt_Arab
Ayacucho Quechua	quy_Latn
Romanian	ron_Latn
Rundi	run_Latn
Russian	rus_Cyrl
Sango	sag_Latn
Sanskrit	san_Deva
Santali	sat_Olck
Sicilian	scn_Latn
Shan	shn_Mymr
Sinhala	sin_Sinh
Slovak	slk_Latn
Slovenian	slv_Latn
Samoan	smo_Latn
Shona	sna_Latn
Sindhi	snd_Arab
Somali	som_Latn
Southern Sotho	sot_Latn
Spanish	spa_Latn
Tosk Albanian	als_Latn
Sardinian	srd_Latn
Serbian	srp_Cyrl
Swati	ssw_Latn
Sundanese	sun_Latn
Swedish	swe_Latn
Swahili	swh_Latn
Silesian	szl_Latn
Tamil	tam_Taml
Tatar	tat_Cyrl
Telugu	tel_Telu
Tajik	tgk_Cyrl
Tagalog	tgl_Latn
Thai	tha_Thai
Tigrinya	tir_Ethi
Tamasheq (Latin script)	taq_Latn
Tamasheq (Tifinagh script)	taq_Tfng
Tok Pisin	tpi_Latn
Tswana	tsn_Latn
Tsonga	tso_Latn
Turkmen	tuk_Latn
Tumbuka	tum_Latn
Turkish	tur_Latn
Twi	twi_Latn
Central Atlas Tamazight	tzm_Tfng
Uyghur	uig_Arab
Ukrainian	ukr_Cyrl
Umbundu	umb_Latn
Urdu	urd_Arab
Northern Uzbek	uzn_Latn
Venetian	vec_Latn
Vietnamese	vie_Latn
Waray	war_Latn
Wolof	wol_Latn
Xhosa	xho_Latn
Eastern Yiddish	ydd_Hebr
Yoruba	yor_Latn
Yue Chinese	yue_Hant
Chinese	zho_Hans
Chinese (Traditional)	zho_Hant
Standard Malay	zsm_Latn
Zulu	zul_Latn'''

codes_as_string = codes_as_string.split('\n')

flores_codes = {}
for code in codes_as_string:
    lang, lang_code = code.split('\t')
    flores_codes[lang] = lang_code
LANGUAGES = list(flores_codes.values())
lang2iso = {x: i for x, i in lang2iso.items() if x in flores_codes.keys()}
iso2flores_code = {iso: flores_codes[lang] for lang, iso in lang2iso.items()}
