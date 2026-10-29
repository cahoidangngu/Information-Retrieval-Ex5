# Per-query evidence for best LTR model on DEV: ranksvm_sampled

### IMPROVED PLAIN-371: "Any update on the scary in vitro avocado data?"
NDCG@10 0.119 -> 0.413 (delta +0.295); relevant total=2, in candidates=2, recall@100=1.000, candidates=100

Entered LTR top-10 (1):
- [rel=2] BM25#11->LTR#7 "Associations of whole-blood fatty acids and dietary intakes with prost"

Left top-10 (1):
- [rel=0] BM25#9->LTR#12 "An update on bioactive plant lignans."

Features with the largest score contribution difference (entered minus left, w*z): matched_terms (+0.32), bm25_title (-0.16), tfidf_cos_body (+0.14)

### IMPROVED PLAIN-430: "Are krill oil supplements better than fish oil capsules?"
NDCG@10 0.615 -> 0.837 (delta +0.223); relevant total=8, in candidates=5, recall@100=0.625, candidates=100

Entered LTR top-10 (3):
- [rel=0] BM25#12->LTR#3 "A 21-day Daniel fast with or without krill oil supplementation improve"
- [rel=2] BM25#15->LTR#5 "Aqueous uptake and sublethal toxicity of p,p'-DDE in non-feeding larva"
- [rel=2] BM25#23->LTR#9 "Organochlorine pesticide air-water exchange and bioconcentration in kr"

Left top-10 (3):
- [rel=0] BM25#6->LTR#14 "Secondary prevention of CHD in UK men: the Diet and Reinfarction Trial"
- [rel=0] BM25#7->LTR#12 "Children's daily exposure to polychlorinated biphenyls from dietary su"
- [rel=0] BM25#10->LTR#11 "Alkylphenols in human milk and their relations to dietary habits in ce"

Features with the largest score contribution difference (entered minus left, w*z): bm25_comb (-0.66), matched_terms (+0.64), lm_dirichlet (+0.35)

### IMPROVED PLAIN-1537: "low-carb diets"
NDCG@10 0.000 -> 0.205 (delta +0.205); relevant total=45, in candidates=8, recall@100=0.178, candidates=100

Entered LTR top-10 (5):
- [rel=1] BM25#11->LTR#3 "Low-Carbohydrate Diets and All-Cause Mortality: A Systematic Review an"
- [rel=0] BM25#20->LTR#7 "The effect of two energy-restricted diets, a low-fructose diet versus "
- [rel=1] BM25#21->LTR#4 "Low-carbohydrate diets and all-cause and cause-specific mortality: Two"
- [rel=0] BM25#25->LTR#6 "Effects of low animal protein or high-fiber diets on urine composition"
- [rel=0] BM25#59->LTR#9 "VEGETARIAN DIETS AND THE INCIDENCE OF CANCER IN A LOW-RISK POPULATION"

Left top-10 (5):
- [rel=0] BM25#4->LTR#33 "Effects of dietary proteins on plasma lipoprotein levels in normal sub"
- [rel=0] BM25#5->LTR#25 "Protein and coronary heart disease: the role of different protein sour"
- [rel=0] BM25#8->LTR#48 "Flaxseed Supplementation (not Dietary Fat Restriction) Reduces Prostat"
- [rel=0] BM25#9->LTR#50 "Flaxseed Supplementation (not Dietary Fat Restriction) Reduces Prostat"
- [rel=0] BM25#10->LTR#13 "Chronic effects of a high-fat diet enriched with virgin olive oil and "

Features with the largest score contribution difference (entered minus left, w*z): bm25_title (+0.15), bm25_comb (-0.11), lm_dirichlet (+0.09)

### IMPROVED PLAIN-207: "Avoiding Cooked Meat Carcinogens"
NDCG@10 0.255 -> 0.460 (delta +0.205); relevant total=55, in candidates=15, recall@100=0.273, candidates=100

Entered LTR top-10 (2):
- [rel=1] BM25#13->LTR#9 "The cooked meat-derived mammary carcinogen 2-amino-1-methyl-6-phenylim"
- [rel=0] BM25#14->LTR#5 "Heterocyclic amines: Mutagens/carcinogens produced during cooking of m"

Left top-10 (2):
- [rel=0] BM25#4->LTR#11 "Effect of cruciferous vegetable consumption on heterocyclic aromatic a"
- [rel=0] BM25#8->LTR#15 "Quantification of the neurotoxic beta-carboline harmane in barbecued/g"

Features with the largest score contribution difference (entered minus left, w*z): bm25_comb (-0.34), lm_dirichlet (+0.32), matched_terms (+0.16)

### IMPROVED PLAIN-1453: "junk food"
NDCG@10 0.305 -> 0.509 (delta +0.204); relevant total=245, in candidates=25, recall@100=0.102, candidates=100

Entered LTR top-10 (5):
- [rel=1] BM25#14->LTR#10 "Food, cooking skills, and health: a literature review."
- [rel=1] BM25#16->LTR#5 "Trends in energy intake among US children by eating location and food "
- [rel=0] BM25#17->LTR#9 "Evaluation of certain food additives and contaminants."
- [rel=1] BM25#36->LTR#8 "Evaluation of certain food additives. Seventy-first report of the Join"
- [rel=1] BM25#46->LTR#7 "Phosphate Additives in Food—a Health Risk"

Left top-10 (5):
- [rel=0] BM25#4->LTR#53 "Consumer knowledge of foodborne microbial hazards and food-handling pr"
- [rel=0] BM25#6->LTR#54 "Background music genre can modulate flavor pleasantness and overall im"
- [rel=0] BM25#7->LTR#35 "Clostridium difficile in food--innocent bystander or serious threat?"
- [rel=0] BM25#8->LTR#37 "Trends in U.S. consumers' safe handling and consumption of food and th"
- [rel=0] BM25#10->LTR#33 "Organically Grown Food Provides Health Benefits to Drosophila melanoga"

Features with the largest score contribution difference (entered minus left, w*z): lm_dirichlet (+0.07), bm25_rank (-0.03), bm25_comb (-0.02)

### IMPROVED PLAIN-280: "Mercury Testing Recommended Before Pregnancy"
NDCG@10 0.322 -> 0.514 (delta +0.191); relevant total=20, in candidates=8, recall@100=0.400, candidates=100

Entered LTR top-10 (3):
- [rel=0] BM25#11->LTR#9 "Speciation of methyl- and ethyl-mercury in hair of breastfed infants a"
- [rel=1] BM25#12->LTR#8 "Monitoring of mercury, arsenic, and lead in traditional Asian herbal p"
- [rel=0] BM25#17->LTR#10 "Mercury in human hair as an indicator of the fish consumption."

Left top-10 (3):
- [rel=0] BM25#4->LTR#20 "Heavy metal content of ayurvedic herbal medicine products."
- [rel=0] BM25#6->LTR#12 "Health food stores' recommendations for nausea and migraines during pr"
- [rel=0] BM25#10->LTR#13 "Detection of toxic heavy metals and pesticide residue in herbal plants"

Features with the largest score contribution difference (entered minus left, w*z): bm25_comb (-0.28), matched_terms (+0.21), lm_dirichlet (+0.20)

### IMPROVED PLAIN-2991: "Cholesterol and Lower Back Pain"
NDCG@10 0.204 -> 0.387 (delta +0.182); relevant total=2, in candidates=1, recall@100=0.500, candidates=100

Entered LTR top-10 (5):
- [rel=0] BM25#14->LTR#3 "Non-specific low back pain."
- [rel=0] BM25#16->LTR#9 "Pain Reduction and Improvement in Range of Motion After Daily Consumpt"
- [rel=0] BM25#18->LTR#7 "Treatment of abdominal pain in irritable bowel syndrome."
- [rel=0] BM25#20->LTR#8 "Aspartame-induced fibromyalgia, an unusual but curable cause of chroni"
- [rel=0] BM25#92->LTR#5 "Faecal retention: a common cause in functional bowel disorders, append"

Left top-10 (5):
- [rel=0] BM25#3->LTR#13 "A pilot study of potassium supplementation in the treatment of hypokal"
- [rel=0] BM25#4->LTR#12 "Triptans (serotonin, 5-HT1B/1D agonists) in migraine: detailed results"
- [rel=0] BM25#8->LTR#11 "Men with Low Serum Cholesterol Have a Lower Risk of High-Grade Prostat"
- [rel=0] BM25#9->LTR#18 "In vitro binding capacities of three dietary fibers and their mixture "
- [rel=0] BM25#10->LTR#46 "Effects of dietary coconut oil, butter and safflower oil on plasma lip"

Features with the largest score contribution difference (entered minus left, w*z): matched_terms (+0.32), coverage_ratio (-0.21), bm25_comb (-0.16)

### IMPROVED PLAIN-2730: "Anti-Angiogenesis: Cutting Off Tumor Supply Lines"
NDCG@10 0.294 -> 0.463 (delta +0.169); relevant total=29, in candidates=10, recall@100=0.345, candidates=100

Entered LTR top-10 (5):
- [rel=2] BM25#14->LTR#4 "Modulation of angiogenesis by dietary phytoconstituents in the prevent"
- [rel=2] BM25#20->LTR#5 "Tumor Angiogenesis as a Target for Dietary Cancer Prevention"
- [rel=0] BM25#27->LTR#6 "Characterization of Campylobacter spp. transferred from naturally cont"
- [rel=0] BM25#32->LTR#7 "Campylobacter transfer from naturally contaminated chicken thighs to c"
- [rel=0] BM25#82->LTR#9 "Final report on the safety assessment of capsicum annuum extract, caps"

Left top-10 (5):
- [rel=0] BM25#3->LTR#21 "Insights into the beneficial effect of caloric/ dietary restriction fo"
- [rel=0] BM25#5->LTR#17 "Evidence for a human-specific mechanism for diet and antibody-mediated"
- [rel=0] BM25#8->LTR#12 "A possible role for the alpha 1-->3 galactosyl epitope and the natural"
- [rel=1] BM25#9->LTR#18 "Antiproliferative effects of apple peel extract against cancer cells."
- [rel=0] BM25#10->LTR#20 "Elevated Levels of Cholesterol-Rich Lipid Rafts in Cancer Cells Are Co"

Features with the largest score contribution difference (entered minus left, w*z): bm25_comb (-0.65), matched_terms (+0.45), lm_dirichlet (+0.41)

### IMPROVED PLAIN-3332: "Alcohol Risks vs. Benefits"
NDCG@10 0.330 -> 0.488 (delta +0.158); relevant total=14, in candidates=9, recall@100=0.643, candidates=100

Entered LTR top-10 (3):
- [rel=1] BM25#11->LTR#9 "Moderate alcohol consumption during adult life, drinking patterns, and"
- [rel=0] BM25#20->LTR#6 "Coffee and its consumption: benefits and risks."
- [rel=0] BM25#26->LTR#7 "Comparison of inhalation of isopropyl alcohol vs promethazine in the t"

Left top-10 (3):
- [rel=0] BM25#6->LTR#17 "Cancer incidence among Danish Seventh-day Adventists and Baptists."
- [rel=0] BM25#8->LTR#11 "Adherence to WCRF/AICR cancer prevention recommendations and risk of p"
- [rel=0] BM25#9->LTR#13 "A probabilistic characterization of the health benefits of reducing me"

Features with the largest score contribution difference (entered minus left, w*z): bm25_comb (-0.25), matched_terms (+0.21), bm25_title (+0.17)

### IMPROVED PLAIN-3141: "Relieving Yourself of Excess Estrogen"
NDCG@10 0.000 -> 0.147 (delta +0.147); relevant total=34, in candidates=5, recall@100=0.147, candidates=100

Entered LTR top-10 (5):
- [rel=2] BM25#11->LTR#4 "Estrogen excretion patterns and plasma levels in vegetarian and omnivo"
- [rel=0] BM25#14->LTR#7 "Dietary lignan intakes and risk of breast cancer by tumor estrogen rec"
- [rel=0] BM25#18->LTR#8 "Public health impact of dietary phosphorus excess on bone and cardiova"
- [rel=0] BM25#20->LTR#9 "Aromatherapy with peppermint, isopropyl alcohol, or placebo is equally"
- [rel=0] BM25#31->LTR#10 "Dietary seaweed modifies estrogen and phytoestrogen metabolism in heal"

Left top-10 (5):
- [rel=0] BM25#4->LTR#21 "DHEA, DHEAS and PCOS."
- [rel=0] BM25#6->LTR#11 "Conjugated equine estrogens and incidence of probable dementia and mil"
- [rel=0] BM25#7->LTR#12 "Muscle soreness and delayed-onset muscle soreness."
- [rel=0] BM25#9->LTR#16 "Mortality in workers employed in pig abattoirs and processing plants."
- [rel=0] BM25#10->LTR#19 "Dietary exposure to xenoestrogens in New Zealand."

Features with the largest score contribution difference (entered minus left, w*z): bm25_title (+0.14), lm_dirichlet (+0.13), bm25_comb (-0.12)

### HARMED PLAIN-3085: "The Difficulty of Arriving at a Vitamin D Recommendation"
NDCG@10 0.484 -> 0.000 (delta -0.484); relevant total=2, in candidates=2, recall@100=1.000, candidates=100

Entered LTR top-10 (4):
- [rel=0] BM25#11->LTR#7 "Towards prevention of vitamin D deficiency and beyond: knowledge gaps "
- [rel=0] BM25#14->LTR#1 "Vitamin D: extraskeletal health."
- [rel=0] BM25#18->LTR#6 "Vitamin D deficiency treated by consuming UVB-irradiated mushrooms"
- [rel=0] BM25#21->LTR#4 "Vitamin D(3) is more potent than vitamin D(2) in humans."

Left top-10 (4):
- [rel=2] BM25#3->LTR#22 "Low Vitamin D Status: Definition, Prevalence, Consequences and Correct"
- [rel=0] BM25#8->LTR#20 "Bioavailability of vitamin D₂ from UV-B-irradiated button mushrooms in"
- [rel=0] BM25#9->LTR#18 "Vitamin D2 Is as Effective as Vitamin D3 in Maintaining Circulating Co"
- [rel=2] BM25#10->LTR#12 "Vitamin D supplement doses and serum 25-hydroxyvitamin D in the range "

Features with the largest score contribution difference (entered minus left, w*z): lm_dirichlet (+0.25), bm25_comb (-0.09), bm25_title (+0.04)

### HARMED PLAIN-3026: "Vitamin C-Enriched Bacon"
NDCG@10 0.315 -> 0.000 (delta -0.315); relevant total=1, in candidates=1, recall@100=1.000, candidates=100

Entered LTR top-10 (2):
- [rel=0] BM25#11->LTR#6 "Hazelnut-enriched diet improves cardiovascular risk biomarkers beyond "
- [rel=0] BM25#13->LTR#9 "Vitamins E and C in the Prevention of Cardiovascular Disease in Men: T"

Left top-10 (2):
- [rel=0] BM25#4->LTR#12 "Effects of dietary proteins on plasma lipoprotein levels in normal sub"
- [rel=2] BM25#8->LTR#14 "Effects of vitamins C and E on N-nitroso compound formation, carcinoge"

Features with the largest score contribution difference (entered minus left, w*z): doc_len (+0.11), bm25_comb (-0.10), bm25_title (+0.06)

### HARMED PLAIN-2460: "Diabetes as a Disease of Fat Toxicity"
NDCG@10 0.399 -> 0.102 (delta -0.296); relevant total=31, in candidates=8, recall@100=0.258, candidates=100

Entered LTR top-10 (6):
- [rel=0] BM25#13->LTR#2 "A low-fat vegan diet and a conventional diabetes diet in the treatment"
- [rel=0] BM25#14->LTR#3 "Changes in nutrient intake and dietary quality among participants with"
- [rel=0] BM25#15->LTR#4 "A prospective study of prepregnancy dietary fat intake and risk of ges"
- [rel=0] BM25#17->LTR#8 "Dietary Fat and Cholesterol and Risk of Cardiovascular Disease in Olde"
- [rel=0] BM25#24->LTR#9 "Vegetarian diet improves insulin resistance and oxidative stress marke"
- [rel=0] BM25#36->LTR#10 "Synergistic toxicity of the environmental neurotoxins methylmercury an"

Left top-10 (6):
- [rel=2] BM25#5->LTR#12 "Banting Memorial Lecture 2012 Reversing the twin cycles of Type 2 diab"
- [rel=0] BM25#6->LTR#16 "Low-carbohydrate diet scores and risk of type 2 diabetes in men"
- [rel=0] BM25#7->LTR#18 "Dynamics of fat cell turnover in humans."
- [rel=0] BM25#8->LTR#14 "Phytochemicals and their impact on adipose tissue inflammation and dia"
- [rel=0] BM25#9->LTR#15 "Sugar, Uric Acid, and the Etiology of Diabetes and Obesity"
- [rel=0] BM25#10->LTR#20 "Hibiscus sabdariffa L. in the treatment of hypertension and hyperlipid"

Features with the largest score contribution difference (entered minus left, w*z): matched_terms (+0.27), coverage_ratio (-0.13), bm25_comb (-0.13)

### HARMED PLAIN-2500: "The Saturated Fat Studies: Buttering Up the Public"
NDCG@10 0.299 -> 0.062 (delta -0.237); relevant total=7, in candidates=3, recall@100=0.429, candidates=100

Entered LTR top-10 (6):
- [rel=1] BM25#11->LTR#4 "Tolerable upper intake levels for trans fat, saturated fat, and choles"
- [rel=0] BM25#12->LTR#6 "Substituting dietary saturated for monounsaturated fat impairs insulin"
- [rel=0] BM25#14->LTR#5 "Dietary saturated fat intake is negatively associated with weight main"
- [rel=0] BM25#18->LTR#7 "Relationship of dietary saturated fatty acids and body habitus to seru"
- [rel=0] BM25#19->LTR#10 "Differential effects of monounsaturated, polyunsaturated and saturated"
- [rel=0] BM25#25->LTR#8 "Effect of low saturated fat diet in early and late cases of multiple s"

Left top-10 (6):
- [rel=0] BM25#4->LTR#12 "A modest proposal for a longitudinal study of dementia prevention (wit"
- [rel=2] BM25#5->LTR#30 "Intake of fatty acids in general populations worldwide does not meet d"
- [rel=0] BM25#6->LTR#11 "Dietary fat and semen quality among men attending a fertility clinic"
- [rel=0] BM25#7->LTR#24 "Dietary intake and the risk of coronary heart disease among the coconu"
- [rel=2] BM25#9->LTR#18 "Recommended dietary reference intakes, nutritional goals and dietary g"
- [rel=0] BM25#10->LTR#15 "Association of dietary fat, vegetables and antioxidant micronutrients "

Features with the largest score contribution difference (entered minus left, w*z): matched_terms (+0.37), bm25_comb (-0.15), coverage_ratio (-0.14)

### HARMED PLAIN-2700: "Heart Disease Starts in Childhood"
NDCG@10 0.215 -> 0.000 (delta -0.215); relevant total=10, in candidates=3, recall@100=0.300, candidates=100

Entered LTR top-10 (4):
- [rel=0] BM25#19->LTR#8 "IQ in childhood and vegetarianism in adulthood: 1970 British cohort st"
- [rel=0] BM25#21->LTR#6 "Childhood soy intake and breast cancer risk in Asian American women."
- [rel=0] BM25#25->LTR#10 "Childhood dairy intake and adult cancer risk: 65-y follow-up of the Bo"
- [rel=0] BM25#27->LTR#7 "A Systematic Review and Meta-analysis of Childhood Leukemia and Parent"

Left top-10 (4):
- [rel=2] BM25#2->LTR#18 "Pathobiological determinants of atherosclerosis in youth risk scores a"
- [rel=0] BM25#6->LTR#19 "The effect of fruit and vegetable intake on risk for coronary heart di"
- [rel=0] BM25#9->LTR#13 "Long-term health and quality-of-life consequences of mass screening fo"
- [rel=0] BM25#10->LTR#14 "A comparative risk assessment of burden of disease and injury attribut"

Features with the largest score contribution difference (entered minus left, w*z): matched_terms (+0.40), bm25_comb (-0.20), coverage_ratio (-0.19)

### HARMED PLAIN-2354: "walnut oil"
NDCG@10 0.553 -> 0.345 (delta -0.208); relevant total=5, in candidates=2, recall@100=0.400, candidates=100

Entered LTR top-10 (1):
- [rel=0] BM25#11->LTR#8 "Long-term walnut supplementation without dietary advice induces favora"

Left top-10 (1):
- [rel=0] BM25#10->LTR#11 "Human cancer cell antiproliferative and antioxidant activities of Jugl"

Features with the largest score contribution difference (entered minus left, w*z): bm25_title (+0.14), tfidf_cos_title (+0.11), doc_len (-0.05)

### HARMED PLAIN-3171: "Convergence of Evidence"
NDCG@10 0.191 -> 0.000 (delta -0.191); relevant total=34, in candidates=8, recall@100=0.235, candidates=100

Entered LTR top-10 (5):
- [rel=0] BM25#13->LTR#3 "Pancreatic cancer: a review of the evidence on causation."
- [rel=0] BM25#17->LTR#8 "Diet in acne: further evidence for the role of nutrient signalling in "
- [rel=0] BM25#23->LTR#10 "Soybean isoflavone exposure does not have feminizing effects on men: a"
- [rel=0] BM25#24->LTR#4 "Can green tea do that? A literature review of the clinical evidence."
- [rel=0] BM25#27->LTR#9 "Homeopathy: what does the "best" evidence tell us?"

Left top-10 (5):
- [rel=1] BM25#2->LTR#39 "The longer-term impacts of Western diet on human cognition and the bra"
- [rel=0] BM25#5->LTR#43 "Probiotics as prevention and treatment for diarrhea."
- [rel=1] BM25#8->LTR#29 "A modest proposal for a longitudinal study of dementia prevention (wit"
- [rel=1] BM25#9->LTR#28 "Risk factors and preventive interventions for Alzheimer disease: state"
- [rel=0] BM25#10->LTR#30 "American Academy of Pediatrics recommendations on the Effects of Early"

Features with the largest score contribution difference (entered minus left, w*z): bm25_title (+0.10), tfidf_cos_title (+0.06), lm_dirichlet (+0.04)

### HARMED PLAIN-2680: "Counteracting the Effects of Dioxins Through Diet"
NDCG@10 0.189 -> 0.000 (delta -0.189); relevant total=14, in candidates=4, recall@100=0.286, candidates=100

Entered LTR top-10 (3):
- [rel=0] BM25#13->LTR#6 "Dioxins, polychlorinated biphenyls, methyl mercury and omega-3 polyuns"
- [rel=0] BM25#15->LTR#8 "Dioxins, dibenzofurans, dioxin-like PCBs, and DDE in U.S. fast food, 1"
- [rel=0] BM25#16->LTR#9 "Intake of dioxins and related compounds from food in the U.S. populati"

Left top-10 (3):
- [rel=2] BM25#5->LTR#13 "Flavones and flavonols at dietary levels inhibit a transformation of a"
- [rel=0] BM25#6->LTR#15 "Could a vegetarian diet reduce exercise-induced oxidative stress? A re"
- [rel=2] BM25#8->LTR#14 "An update on the dietary ligands of the AhR."

Features with the largest score contribution difference (entered minus left, w*z): matched_terms (+0.43), bm25_comb (-0.32), coverage_ratio (-0.21)

### HARMED PLAIN-2540: "Does Cholesterol Size Matter?"
NDCG@10 0.253 -> 0.073 (delta -0.180); relevant total=21, in candidates=9, recall@100=0.429, candidates=100

Entered LTR top-10 (2):
- [rel=0] BM25#16->LTR#4 "Human gut microbiota: does diet matter?"
- [rel=0] BM25#18->LTR#9 "Abnormal neuronal migration, deranged cerebral cortical organization, "

Left top-10 (2):
- [rel=0] BM25#6->LTR#23 "Update on the biological effects of ionizing radiation, relative dose "
- [rel=1] BM25#9->LTR#13 "Cheese intake in large amounts lowers LDL-cholesterol concentrations c"

Features with the largest score contribution difference (entered minus left, w*z): bm25_comb (-0.16), matched_terms (+0.16), bm25_title (+0.16)

### HARMED PLAIN-102: "Stopping Heart Disease in Childhood"
NDCG@10 0.149 -> 0.000 (delta -0.149); relevant total=24, in candidates=1, recall@100=0.042, candidates=100

Entered LTR top-10 (3):
- [rel=0] BM25#12->LTR#9 "Plant foods and plant-based diets: protective against childhood obesit"
- [rel=0] BM25#13->LTR#6 "The burden of childhood asthma"
- [rel=0] BM25#32->LTR#10 "A Systematic Review and Meta-analysis of Childhood Leukemia and Parent"

Left top-10 (3):
- [rel=2] BM25#5->LTR#22 "Pathobiological determinants of atherosclerosis in youth risk scores a"
- [rel=0] BM25#6->LTR#12 "Risks and benefits of estrogen plus progestin in healthy postmenopausa"
- [rel=0] BM25#8->LTR#21 "Differences in metal content of the heart muscle in death from ischemi"

Features with the largest score contribution difference (entered minus left, w*z): lm_dirichlet (+0.32), matched_terms (+0.32), bm25_comb (-0.24)
