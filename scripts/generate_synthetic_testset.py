#!/usr/bin/env python3
"""
Generate a synthetic gold test set with 100% certain labels.

Sentences written from scratch — not from any corpus, not from training data.
No borderline cases. Every sentence is either a clear, explicit biotic
interaction or clearly NOT an interaction (co-occurrence, methodology, etc.).

50 positives: diverse interaction types (predation, parasitism, pollination,
              herbivory, dispersal, vector, mutualism, parasitoidism, etc.)
50 negatives: co-occurrence, methodology, geographic overlap, taxonomy,
              background review, negated interactions, experimental setup.
"""

import pandas as pd
from pathlib import Path

OUT = Path("/path/to/MetaP/classifier/data/evaluation/synthetic_gold_100.tsv")

# ── POSITIVES (label=1) ────────────────────────────────────────────────────
# Each sentence unambiguously describes an actual biotic interaction occurring.

positives = [
    # PREDATION / eats / preysOn
    ("Grey wolves (Canis lupus) hunted and killed elk (Cervus canadensis) calves in Yellowstone National Park.", "predation"),
    ("The great white shark (Carcharodon carcharias) attacked and consumed a Cape fur seal (Arctocephalus pusillus) near the surface.", "predation"),
    ("Barn owls (Tyto alba) captured and ate house mice (Mus musculus) in agricultural fields.", "predation"),
    ("The African lion (Panthera leo) preyed on wildebeest (Connochaetes taurinus) during the dry season migration.", "predation"),
    ("Eurasian sparrowhawks (Accipiter nisus) seized and killed blue tits (Cyanistes caeruleus) at garden feeders.", "predation"),
    ("The water spider (Dolomedes fimbriatus) caught and consumed tadpoles (Rana temporaria) from pond surface.", "predation"),
    ("Peregrine falcons (Falco peregrinus) stooped on and killed common starlings (Sturnus vulgaris) mid-flight.", "predation"),

    # PARASITISM (internal)
    ("Plasmodium falciparum infected red blood cells of Homo sapiens and multiplied within them, causing malaria.", "parasitism"),
    ("Toxoplasma gondii formed cysts in the brain tissue of Mus musculus following oral ingestion of oocysts.", "parasitism"),
    ("The tapeworm Taenia solium established itself in the intestine of Sus scrofa and absorbed nutrients from the host gut.", "parasitism"),
    ("Ascaris lumbricoides larvae migrated through the lungs of infected children before maturing in the small intestine.", "parasitism"),
    ("Ophiocordyceps unilateralis infected Camponotus leonardi ants, manipulating their behaviour before killing them.", "parasitism"),
    ("Varroa destructor fed on the haemolymph of Apis mellifera pupae within sealed brood cells.", "ectoparasitism"),

    # PARASITOIDISM
    ("Cotesia glomerata larvae hatched inside Pieris brassicae caterpillars, feeding on the host tissues before pupating externally.", "parasitoidism"),
    ("The ichneumon wasp Venturia canescens deposited eggs inside Ephestia kuehniella larvae, which were subsequently killed by the developing parasitoid.", "parasitoidism"),
    ("Nasonia vitripennis pupated within Calliphora vicina puparia after the wasp larvae consumed the host fly.", "parasitoidism"),

    # HERBIVORY
    ("Manduca sexta caterpillars consumed Nicotiana attenuata leaves, removing up to 80% of the leaf area.", "herbivory"),
    ("The mountain pine beetle (Dendroctonus ponderosae) bored into Pinus contorta phloem and fed on inner bark tissue.", "herbivory"),
    ("Locusta migratoria swarms defoliated Sorghum bicolor fields, stripping leaves and stems within hours.", "herbivory"),
    ("Sitobion avenae aphids fed on Triticum aestivum phloem sap, stunting plant growth and reducing grain yield.", "herbivory"),
    ("Ips typographus larvae fed under the bark of Picea abies, girdling the trees and causing mortality.", "herbivory"),

    # POLLINATION
    ("Apis mellifera workers transferred pollen from Malus domestica anthers to stigmas while foraging for nectar.", "pollination"),
    ("Bombus terrestris queens visited Lavandula angustifolia flowers and deposited pollen on receptive stigmas.", "pollination"),
    ("The hawkmoth Manduca sexta hovered at Nicotiana alata flowers and pollinated them while feeding on nectar.", "pollination"),
    ("Eulaema meriana collected and deposited pollen on Euglossa orchid flowers during fragrance-gathering visits.", "pollination"),
    ("Datura wrightii was exclusively pollinated by Manduca quinquemaculata moths, which inserted their proboscis into the flowers at night.", "pollination"),

    # SEED DISPERSAL
    ("Eurasian jays (Garrulus glandarius) cached Quercus robur acorns up to 2 km from the parent tree, germinating the following spring.", "seed_dispersal"),
    ("Fruit bats (Pteropus alecto) ingested Ficus fruits and defecated the seeds at roost sites, dispersing them across the landscape.", "seed_dispersal"),
    ("Tapirs (Tapirus terrestris) consumed Attalea butyracea palm fruits and deposited viable seeds in their dung.", "seed_dispersal"),
    ("Elephants (Loxodonta africana) swallowed Acacia tortilis pods and excreted intact seeds up to 65 km away.", "seed_dispersal"),

    # VECTOR / DISEASE TRANSMISSION
    ("Aedes aegypti transmitted dengue virus to Homo sapiens during a blood meal, injecting infected saliva.", "vectorOf"),
    ("Ixodes scapularis transferred Borrelia burgdorferi to white-footed mice (Peromyscus leucopus) while feeding.", "vectorOf"),
    ("Anopheles gambiae inoculated Plasmodium falciparum sporozoites into the bloodstream of human hosts during feeding.", "vectorOf"),
    ("Culex pipiens transmitted West Nile virus to Corvus brachyrhynchos during blood feeding.", "vectorOf"),
    ("Triatoma infestans deposited Trypanosoma cruzi in its faeces near the bite wound of Homo sapiens, causing Chagas disease.", "vectorOf"),

    # NITROGEN FIXATION / MUTUALISM
    ("Rhizobium leguminosarum colonised root nodules of Pisum sativum and fixed atmospheric nitrogen for the plant.", "mutualism"),
    ("Frankia alni formed nitrogen-fixing nodules on Alnus glutinosa roots, supplying the tree with combined nitrogen.", "mutualism"),
    ("Glomus mosseae formed arbuscular mycorrhizal associations with Zea mays roots, enhancing phosphorus uptake.", "mutualism"),
    ("Symbiodinium sp. photosynthesised within the tissues of Acropora millepora coral, providing the host with fixed carbon.", "mutualism"),
    ("Wolbachia pipientis provided nutritional supplements to Cimex lectularius and protected the bed bug from vitamin B deficiencies.", "mutualism"),

    # COMPETITION
    ("Rattus norvegicus outcompeted Rattus rattus for grain stores in barns, displacing the black rat from occupied sites.", "competition"),
    ("Invasive Phytophthora cinnamomi suppressed Quercus suber regeneration by competing for soil water and nutrients.", "competition"),

    # KLEPTOPARASITISM
    ("Stercorarius parasiticus chased Arctic terns (Sterna paradisaea) until they dropped their fish, which the skua then consumed.", "kleptoparasitism"),
    ("Hyenas (Crocuta crocuta) drove lions (Panthera leo) from a zebra carcass and consumed the kill themselves.", "kleptoparasitism"),

    # BROOD PARASITISM
    ("The common cuckoo (Cuculus canorus) laid an egg in the nest of a reed warbler (Acrocephalus scirpaceus), whose chick was subsequently evicted by the cuckoo nestling.", "brood_parasitism"),

    # CLEANING SYMBIOSIS
    ("Labroides dimidiatus cleaned ectoparasites from the gills and skin of Epinephelus fasciatus at a coral reef cleaning station.", "cleaning_symbiosis"),

    # MYRMECOPHILY / ANT MUTUALISM
    ("Lycaena arion caterpillars were carried into Myrmica sabuleti nests and fed on ant larvae for ten months.", "myrmecophily"),

    # PHORESY
    ("Macrocheles muscaedomesticae mites attached to Musca domestica and were transported to new dung patches.", "phoresy"),

    # ENDOPHYTE / PLANT-FUNGUS MUTUALISM
    ("Neotyphodium coenophialum grew endophytically within Festuca arundinacea tissues and produced alkaloids that deterred insect herbivores.", "mutualism"),

    # EPIPHYTE / HOST PLANT
    ("The strangler fig (Ficus aurea) germinated on Sabal palmetto and sent roots down to the soil, eventually overgrowing and killing the host palm.", "parasitism"),
]

# ── NEGATIVES (label=0) ────────────────────────────────────────────────────
# Co-occurrence, methodology, background, negated interactions, taxonomy, etc.
# None of these describe an actual ongoing biotic interaction.

negatives = [
    # CO-OCCURRENCE (same habitat, no interaction stated)
    ("Canis lupus and Cervus canadensis both inhabit the boreal forests of North America.", "cooccurrence"),
    ("Plasmodium falciparum and Homo sapiens are found in sub-Saharan Africa.", "cooccurrence"),
    ("Apis mellifera and Malus domestica are commonly found in European orchards.", "cooccurrence"),
    ("Quercus robur and Garrulus glandarius share overlapping geographic ranges across temperate Europe.", "cooccurrence"),
    ("Aedes aegypti and Homo sapiens populations overlap extensively in tropical urban environments.", "cooccurrence"),
    ("Several bat species and Ficus trees co-occur in the same tropical forest patches.", "cooccurrence"),

    # METHODOLOGY / EXPERIMENTAL SETUP (no actual interaction happening)
    ("We exposed Mus musculus to Plasmodium berghei-infected red blood cells to measure cytokine responses.", "methodology"),
    ("Apis mellifera colonies were kept in wooden hives adjacent to Malus domestica orchards for the duration of the experiment.", "methodology"),
    ("Cotesia glomerata parasitoid wasps were maintained in the laboratory on Pieris brassicae for colony maintenance.", "methodology"),
    ("Toxoplasma gondii tachyzoites were cultured in HeLa cell monolayers at 37°C.", "methodology"),
    ("Manduca sexta eggs were placed on Nicotiana attenuata leaves to test feeding preference.", "methodology"),
    ("Rhizobium leguminosarum cultures were grown on yeast mannitol agar at 28°C before inoculation trials.", "methodology"),
    ("Anopheles gambiae mosquitoes were reared in insectary conditions and blood-fed on sheep to maintain the colony.", "methodology"),
    ("Peromyscus leucopus tissue samples were collected from trapped individuals and stored at -80°C.", "methodology"),

    # BACKGROUND / REVIEW STATEMENTS (general knowledge, not an event)
    ("Wolves are known to be predators of elk in many ecosystems.", "background"),
    ("Plasmodium falciparum is the most lethal malaria parasite affecting humans.", "background"),
    ("Bees are well-established pollinators of many agricultural crops worldwide.", "background"),
    ("The relationship between mycorrhizal fungi and plant roots has been studied extensively.", "background"),
    ("Varroa destructor has been identified as a major threat to honeybee populations globally.", "background"),
    ("Dengue virus transmission involves mosquito vectors of the genus Aedes.", "background"),
    ("Host-parasite relationships are a major driver of evolutionary arms races.", "background"),
    ("Seed dispersal by vertebrates plays an important role in plant community dynamics.", "background"),

    # NEGATED INTERACTIONS
    ("Wolves did not prey on deer in this enclosed study area during the observation period.", "negated"),
    ("No transmission of Borrelia burgdorferi from Ixodes ticks to deer was observed in this experiment.", "negated"),
    ("Apis mellifera did not visit Nicotiana attenuata flowers under field conditions.", "negated"),
    ("The parasitoid wasp was unable to parasitise Manduca sexta larvae beyond the third instar.", "negated"),

    # GEOGRAPHIC / TAXONOMIC STATEMENTS
    ("Canis lupus is a member of the family Canidae and is the largest wild canid.", "taxonomy"),
    ("Plasmodium falciparum belongs to the phylum Apicomplexa and genus Plasmodium.", "taxonomy"),
    ("The distribution of Quercus robur extends from western Europe to the Ural Mountains.", "taxonomy"),
    ("Corvus monedula has been classified as a species of least concern by the IUCN.", "taxonomy"),
    ("Apis mellifera was introduced to North America in the seventeenth century.", "taxonomy"),

    # POPULATION / ECOLOGICAL MONITORING (no specific interaction)
    ("Elk populations in Yellowstone increased by 15% between 2010 and 2015.", "monitoring"),
    ("Honeybee colony losses exceeded 30% in Europe during the winter of 2014.", "monitoring"),
    ("Wolf pack sizes in Yellowstone ranged from 3 to 12 individuals during the study period.", "monitoring"),
    ("The abundance of Aedes aegypti was monitored weekly using ovitraps in urban areas.", "monitoring"),
    ("Picea abies stand density declined significantly in plots affected by bark beetle outbreaks.", "monitoring"),

    # PHYLOGENETICS / COMPARATIVE BIOLOGY
    ("A phylogenetic analysis placed Ophiocordyceps unilateralis within the family Ophiocordycipitaceae.", "phylogenetics"),
    ("Borrelia burgdorferi and Borrelia afzelii are both members of the Borrelia burgdorferi sensu lato complex.", "phylogenetics"),
    ("Genomic comparison of Plasmodium falciparum and Plasmodium vivax revealed significant synteny.", "phylogenetics"),

    # CHEMICAL / PHYSIOLOGICAL (species mentioned but no ecological interaction)
    ("Nicotiana attenuata produces trypsin protease inhibitors in response to wounding.", "physiology"),
    ("Canis lupus has a highly developed olfactory system with over 300 million scent receptors.", "physiology"),
    ("Manduca sexta larvae produce digestive enzymes that break down plant cell walls.", "physiology"),
    ("Apis mellifera workers synthesise royal jelly from hypopharyngeal gland secretions.", "physiology"),

    # CLIMATE / ENVIRONMENTAL CONTEXT (no interaction)
    ("Temperature increases of 2°C are projected to shift the range of Aedes aegypti northward.", "environmental"),
    ("Habitat fragmentation has reduced connectivity between wolf populations in Europe.", "environmental"),
    ("Drought conditions in 2018 reduced floral resources for pollinators across Mediterranean Europe.", "environmental"),

    # GENOMICS / MOLECULAR (no ecological interaction)
    ("The genome of Plasmodium falciparum was sequenced and found to contain approximately 23 megabases.", "genomics"),
    ("Wolbachia pipientis carries a bacteriophage (WO) integrated into its genome.", "genomics"),
    ("The Varroa destructor genome encodes a suite of salivary proteins expressed during feeding.", "genomics"),
    ("Whole-genome sequencing of Cotesia glomerata revealed expansions in venom gene families.", "genomics"),
]

# ── Assemble ───────────────────────────────────────────────────────────────
assert len(positives) == 50, f"Expected 50 positives, got {len(positives)}"
assert len(negatives) == 50, f"Expected 50 negatives, got {len(negatives)}"

rows = []
for text, itype in positives:
    rows.append({"text": text.strip(), "label": 1, "interaction_type": itype, "source": "synthetic_gold"})
for text, itype in negatives:
    rows.append({"text": text.strip(), "label": 0, "interaction_type": itype, "source": "synthetic_gold"})

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, sep="\t", index=False)

print(f"Saved {len(df)} sentences to {OUT}")
print(f"  Positives: {(df['label']==1).sum()}")
print(f"  Negatives: {(df['label']==0).sum()}")
print(f"  Interaction types (pos): {df[df['label']==1]['interaction_type'].value_counts().to_dict()}")
print(f"  Negative types: {df[df['label']==0]['interaction_type'].value_counts().to_dict()}")
