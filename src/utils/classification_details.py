#just a record of the claude code that provides classification_details.csv

#!/usr/bin/env python3
"""
Species Interaction Classifier

This script classifies sentences to determine if they describe real biological 
interactions between species, or if they are false positives where species and 
interaction terms appear together but don't describe an actual interaction.

Input: 
    - positive_no_eat.csv: sentences containing 2 species + 1 interaction term
    - interaction_dict.csv: dictionary of interaction terms

Output:
    - true_positives.csv: sentences with real species interactions
    - false_positives.csv: sentences without real interactions
    - classification_details.csv: all sentences with classification, confidence, and reason
"""

import pandas as pd
import re


# ============================================================================
# TRUE POSITIVE PATTERNS - Evidence of real biological interaction
# ============================================================================

# Infection/Disease causation
INFECTION_PATTERNS = [
    r'\binfect(?:s|ed|ing|ion|ious)?\b.*\b(?:with|by|of|in|from)\b',
    r'\b(?:with|by)\b.*\binfect',
    r'\bcaus(?:e[sd]?|ing)\s+(?:\w+\s+)?(?:disease|infection|illness|syndrome)',
    r'\b(?:disease|infection|illness)\s+(?:caused|due)\s+(?:by|to)\b',
    r'\bpathogen(?:s|ic)?\s+(?:of|in|for)\b',
    r'\b(?:agent|organism)\s+(?:of|causing)\s+\w+\s+(?:disease|infection)',
]

# Host-parasite/vector relationships
HOST_PARASITE_PATTERNS = [
    r'\bhost(?:s|ed)?\s+(?:of|for|to)\b',
    r'\b(?:intermediate|definitive|final|paratenic|reservoir)\s+host',
    r'\bparasit(?:e[sd]?|ic|ism|iz)',
    r'\bvector(?:s|ed)?\s+(?:of|for|by)\b',
    r'\btransmit(?:s|ted|ting)?\s+(?:by|to|from)\b',
    r'\b(?:transmitted|spread|carried)\s+by\b',
    r'\bcarrier(?:s)?\s+(?:of|for)\b',
]

# Predation/feeding
PREDATION_PATTERNS = [
    r'\bprey(?:s|ed)?\s+(?:on|upon)\b',
    r'\bpredator(?:s|y)?\s+(?:of|on)\b',
    r'\bpredation\s+(?:on|of|by)\b',
    r'\bfeed(?:s|ing)?\s+(?:on|upon)\b',
    r'\bfed\s+(?:on|upon|to)\b',
    r'\bgraze[sd]?\s+(?:on|upon)\b',
    r'\bgrazing\s+(?:on|by)\b',
]

# Colonization/invasion/infestation
COLONIZATION_PATTERNS = [
    r'\bcoloniz(?:e[sd]?|ation|ing)\b',
    r'\binvad(?:e[sd]?|ing)\b',
    r'\binvasion\s+(?:of|by|in)\b',
    r'\binfest(?:s|ed|ing|ation)?\b',
]

# Symbiosis/mutualism
SYMBIOSIS_PATTERNS = [
    r'\bsymbio(?:nt|sis|tic)\b',
    r'\bmutual(?:ist|ism|istic)\b',
    r'\bcommensal(?:ism)?\b',
    r'\bendosymbio',
]

# Pollination
POLLINATION_PATTERNS = [
    r'\bpollinat(?:e[sd]?|ing|ion|or)\b',
    r'\bflower(?:s)?\s+visited\s+by\b',
    r'\bvisit(?:s|ed|ing)?\s+flower',
]

# Experimental interaction context
EXPERIMENTAL_PATTERNS = [
    r'\b(?:mice|rats?|rabbits?|hamsters?|monkeys?|dogs?|guinea\s*pigs?)\s+(?:infected|inoculated|challenged|exposed)\s+(?:with|to)\b',
    r'\b(?:inoculated?|challenged?)\s+(?:with|into)\s+\w+',
    r'\bexperimental\s+(?:infection|challenge)\b',
    r'\b(?:given|administered|injected)\s+\w+\s+(?:into|to)\s+(?:mice|rats?|rabbits?)',
]

# Life cycle in host
LIFECYCLE_PATTERNS = [
    r'\b(?:larvae?|cercariae?|metacercariae?|miracidia|oocysts?|sporozoites?|cysts?)\s+(?:in|of|from)\s+\w+',
    r'\b(?:develop(?:s|ed|ing)?|matures?|stages?)\s+in\s+\w+\s+(?:host|tissue|organ)',
    r'\blife\s*cycle\s+(?:in|of)\b',
]

# Immune response to organism
IMMUNE_PATTERNS = [
    r'\b(?:antibod(?:y|ies)|antigen)\s+(?:to|against|from)\s+\w+',
    r'\bimmun(?:e|ity|ization)\s+(?:to|against|response)\b',
    r'\bresist(?:ant|ance)\s+to\s+\w+\s+(?:infection|disease|organism)',
    r'\bvaccinat(?:e[sd]?|ion)\s+(?:against|with)\b',
]

# Combine all TRUE POSITIVE patterns
ALL_TRUE_POSITIVE_PATTERNS = (
    INFECTION_PATTERNS + HOST_PARASITE_PATTERNS + PREDATION_PATTERNS + 
    COLONIZATION_PATTERNS + SYMBIOSIS_PATTERNS + POLLINATION_PATTERNS +
    EXPERIMENTAL_PATTERNS + LIFECYCLE_PATTERNS + IMMUNE_PATTERNS
)


# ============================================================================
# FALSE POSITIVE PATTERNS - Evidence of NO real interaction
# ============================================================================

# Source/origin (species as material source, not interaction partner)
SOURCE_PATTERNS = [
    r'\b(?:purified|isolated|extracted|obtained|derived|prepared)\s+from\s+\w+\s+(?:tissue|serum|blood|plasma|liver|kidney|brain|muscle|cell|organ)',
    r'\b(?:from|of)\s+\w+\s+(?:liver|kidney|brain|muscle|tissue|serum|plasma|blood|cells?)\b',
    r'\b(?:enzyme|protein|antibody|antigen|dna|rna|gene)\s+(?:from|of)\s+\w+\b',
]

# Comparative/similarity statements (not interaction)
COMPARATIVE_PATTERNS = [
    r'\bsimilar\s+to\s+(?:that|those)\s+(?:of|from|in)\b',
    r'\bdifferent\s+from\s+(?:that|those)\s+(?:of|from|in)\b',
    r'\bsame\s+as\s+(?:that|those|in)\b',
    r'\banalogous\s+to\b',
    r'\bhomologous\s+(?:to|with)\b',
    r'\bidentical\s+to\s+(?:that|those)\b',
    r'\bresembles?\s+(?:that|those)\b',
    r'\bcompared?\s+(?:to|with)\s+(?:that|those)\b',
]

# Molecular/biochemical characterization (not ecological interaction)
MOLECULAR_PATTERNS = [
    r'\bamino\s+acid\s+(?:sequence|composition|content)\b',
    r'\bnucleotide\s+sequence\b',
    r'\bprotein\s+(?:sequence|structure)\b',
    r'\bmolecular\s+weight\b',
    r'\bisoelectric\s+point\b',
    r'\belectrophoretic\s+(?:mobility|pattern)\b',
    r'\bchromatograph(?:y|ic)\b',
    r'\bsequence\s+(?:of|was|were|analysis|determination|homology)\b',
]

# Cloning/expression systems (E. coli as tool, not interaction partner)
CLONING_PATTERNS = [
    r'\bcloned?\s+(?:in|into)\s+(?:e\.?\s*coli|escherichia)\b',
    r'\bexpressed?\s+(?:in|by)\s+(?:e\.?\s*coli|escherichia)\b',
    r'\btransformed?\s+(?:into|with)\s+(?:e\.?\s*coli|escherichia)\b',
    r'\b(?:plasmid|vector)\s+(?:in|into)\s+(?:e\.?\s*coli|escherichia)\b',
]

# Taxonomic/classification (not interaction)
TAXONOMIC_PATTERNS = [
    r'\bbelongs?\s+to\s+(?:the\s+)?(?:genus|family|order|class)\b',
    r'\bclassified\s+(?:as|in|with)\b',
    r'\bphylogenetic(?:ally)?\s+(?:related|relationship|position)\b',
    r'\btaxonom(?:y|ic|ically)\b',
]

# Combine all FALSE POSITIVE patterns
ALL_FALSE_POSITIVE_PATTERNS = (
    SOURCE_PATTERNS + COMPARATIVE_PATTERNS + MOLECULAR_PATTERNS +
    CLONING_PATTERNS + TAXONOMIC_PATTERNS
)


def count_pattern_matches(text, patterns):
    """
    Count how many patterns from a list match in the text.
    
    Args:
        text: The sentence to analyze
        patterns: List of regex patterns to check
        
    Returns:
        Tuple of (count, list of matched patterns)
    """
    text_lower = text.lower()
    count = 0
    matched = []
    for pattern in patterns:
        if re.search(pattern, text_lower):
            count += 1
            matched.append(pattern[:40])  # Store truncated pattern for debugging
    return count, matched


def classify_sentence(text):
    """
    Classify a sentence as true positive (real interaction) or false positive.
    
    Args:
        text: The sentence to classify
        
    Returns:
        Tuple of (classification, confidence, reason)
        - classification: 'true_positive' or 'false_positive'
        - confidence: 'high', 'medium', or 'low'
        - reason: explanation for the classification
    """
    # Handle empty or invalid input
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 15:
        return 'false_positive', 'high', 'too_short'
    
    text = str(text).strip()
    text_lower = text.lower()
    
    # Count pattern matches
    tp_count, tp_matched = count_pattern_matches(text, ALL_TRUE_POSITIVE_PATTERNS)
    fp_count, fp_matched = count_pattern_matches(text, ALL_FALSE_POSITIVE_PATTERNS)
    
    # ==================== DECISION LOGIC ====================
    
    # Strong TRUE POSITIVE: Multiple TP patterns, no FP
    if tp_count >= 2 and fp_count == 0:
        return 'true_positive', 'high', f'strong_interaction_evidence ({tp_count} patterns)'
    
    # Strong TRUE POSITIVE: At least one TP pattern, no FP
    if tp_count >= 1 and fp_count == 0:
        return 'true_positive', 'high', 'clear_interaction_pattern'
    
    # Strong FALSE POSITIVE: Multiple FP patterns, no TP
    if fp_count >= 2 and tp_count == 0:
        return 'false_positive', 'high', f'methodology_focused ({fp_count} patterns)'
    
    # Moderate FALSE POSITIVE: One FP pattern, no TP
    if fp_count >= 1 and tp_count == 0:
        return 'false_positive', 'medium', 'methodology_pattern_present'
    
    # CONFLICT: Both TP and FP patterns present
    if tp_count >= 1 and fp_count >= 1:
        if tp_count > fp_count:
            return 'true_positive', 'medium', 'interaction_despite_methodology'
        elif fp_count > tp_count:
            return 'false_positive', 'medium', 'methodology_despite_interaction'
        else:
            # Equal counts - check for strong interaction patterns
            strong_tp = any(re.search(p, text_lower) for p in 
                          INFECTION_PATTERNS + HOST_PARASITE_PATTERNS + PREDATION_PATTERNS)
            if strong_tp:
                return 'true_positive', 'low', 'strong_interaction_in_mixed_context'
            else:
                return 'false_positive', 'low', 'unclear_mixed_context'
    
    # ==================== HEURISTICS (no strong patterns) ====================
    
    # Check for disease/causation language
    if re.search(r'\b(?:disease|infection|illness|syndrome)\s+(?:in|of|caused|due)\b', text_lower):
        return 'true_positive', 'medium', 'disease_context'
    
    # Check for experimental organism context
    if re.search(r'\b(?:mice|rats?|rabbits?|hamsters?)\s+(?:were|was|are|is)\s+(?:used|given|treated|injected|challenged)', text_lower):
        return 'true_positive', 'medium', 'experimental_organism_context'
    
    # Check for organism-in-organism context
    if re.search(r'\bin\s+(?:infected|diseased|healthy)\s+(?:mice|rats?|humans?|patients?|animals?)', text_lower):
        return 'true_positive', 'medium', 'organism_in_organism'
    
    # Check for "due to" or "caused by" with organism
    if re.search(r'\b(?:due\s+to|caused\s+by)\s+\w+', text_lower):
        return 'true_positive', 'low', 'causation_language'
    
    # Check for biological process words that suggest interaction
    bio_interaction_words = ['phagocytos', 'endocytos', 'virulence', 'pathogenic', 'toxin', 'lethal', 'mortality']
    if any(word in text_lower for word in bio_interaction_words):
        return 'true_positive', 'low', 'biological_interaction_term'
    
    # Check if it's primarily about molecular properties
    mol_words = ['sequence', 'amino acid', 'nucleotide', 'molecular', 'purification', 'isolation', 'chromatography']
    mol_count = sum(1 for word in mol_words if word in text_lower)
    if mol_count >= 2:
        return 'false_positive', 'low', 'molecular_focus'
    
    # Check for pure comparison without interaction
    if re.search(r'\b(?:similar|different|same|compared|identical|analogous)\b', text_lower):
        if not re.search(r'\binfect|\bhost|\bparasite|\bvector|\bprey|\bfeed', text_lower):
            return 'false_positive', 'low', 'comparative_statement'
    
    # Check for interaction context hints
    interaction_hints = ['in mice', 'in rats', 'in rabbit', 'in human', 'in patient', 
                        'to mice', 'to rats', 'from mice', 'from rats',
                        'against', 'response to', 'effect on', 'effect of']
    if any(hint in text_lower for hint in interaction_hints):
        return 'true_positive', 'low', 'interaction_context_hint'
    
    # Default: assume interaction since original matching found species + interaction term
    return 'true_positive', 'low', 'default_assumed_interaction'


def main():
    """Main function to run the classification."""
    
    # Load input data
    print("Loading data...")
    df = pd.read_csv('positive_no_eat.csv')
    print(f"Total sentences to classify: {len(df)}")
    
    # Process all sentences
    print("\nClassifying sentences...")
    results = []
    for idx, row in df.iterrows():
        text = row['passage']
        classification, confidence, reason = classify_sentence(text)
        results.append({
            'passage': text,
            'classification': classification,
            'confidence': confidence,
            'reason': reason
        })
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1} sentences...")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    print(f"\nTotal sentences: {len(results_df)}")
    print(f"\nBy classification:")
    print(results_df['classification'].value_counts())
    print(f"\nBy confidence:")
    print(results_df.groupby(['classification', 'confidence']).size().unstack(fill_value=0))
    print(f"\nTop reasons:")
    print(results_df['reason'].value_counts().head(15))
    
    # Split into true and false positives
    true_positives = results_df[results_df['classification'] == 'true_positive'][['passage']]
    false_positives = results_df[results_df['classification'] == 'false_positive'][['passage']]
    
    print(f"\n\nFINAL COUNTS:")
    print(f"  TRUE POSITIVES:  {len(true_positives)}")
    print(f"  FALSE POSITIVES: {len(false_positives)}")
    
    # Save output files
    print("\nSaving files...")
    true_positives.to_csv('true_positives.csv', index=False)
    false_positives.to_csv('false_positives.csv', index=False)
    results_df.to_csv('classification_details.csv', index=False)
    
    print("\nDone! Output files:")
    print("  - true_positives.csv")
    print("  - false_positives.csv")
    print("  - classification_details.csv")


if __name__ == '__main__':
    main()