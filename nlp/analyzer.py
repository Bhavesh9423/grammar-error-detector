"""
Linguistic Analysis Module for NLP Academic Demonstration.

Implements core NLP tasks:
1. Sentence Segmentation (Sentence Tokenization)
2. Word & Punctuation Tokenization
3. Part-of-Speech (POS) Tagging (Penn Treebank tagset)
4. Lemmatization using WordNet
5. Lexical Diversity & Syntactic Metrics
"""

import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Initialize lemmatizer singleton
lemmatizer = WordNetLemmatizer()

# Penn Treebank POS tag descriptions and category groupings
POS_TAG_MAP = {
    'CC': {'desc': 'Coordinating conjunction', 'cat': 'Conjunction', 'color': 'tag-conj'},
    'CD': {'desc': 'Cardinal number', 'cat': 'Numeral', 'color': 'tag-num'},
    'DT': {'desc': 'Determiner / Article', 'cat': 'Determiner', 'color': 'tag-det'},
    'EX': {'desc': 'Existential there', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'FW': {'desc': 'Foreign word', 'cat': 'Other', 'color': 'tag-other'},
    'IN': {'desc': 'Preposition / Subordinating conjunction', 'cat': 'Preposition', 'color': 'tag-prep'},
    'JJ': {'desc': 'Adjective (base)', 'cat': 'Adjective', 'color': 'tag-adj'},
    'JJR': {'desc': 'Adjective, comparative', 'cat': 'Adjective', 'color': 'tag-adj'},
    'JJS': {'desc': 'Adjective, superlative', 'cat': 'Adjective', 'color': 'tag-adj'},
    'LS': {'desc': 'List item marker', 'cat': 'Other', 'color': 'tag-other'},
    'MD': {'desc': 'Modal auxiliary verb', 'cat': 'Verb', 'color': 'tag-verb'},
    'NN': {'desc': 'Noun, singular or mass', 'cat': 'Noun', 'color': 'tag-noun'},
    'NNS': {'desc': 'Noun, plural', 'cat': 'Noun', 'color': 'tag-noun'},
    'NNP': {'desc': 'Proper noun, singular', 'cat': 'Noun', 'color': 'tag-noun'},
    'NNPS': {'desc': 'Proper noun, plural', 'cat': 'Noun', 'color': 'tag-noun'},
    'PDT': {'desc': 'Predeterminer', 'cat': 'Determiner', 'color': 'tag-det'},
    'POS': {'desc': 'Possessive ending', 'cat': 'Punctuation', 'color': 'tag-punct'},
    'PRP': {'desc': 'Personal pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'PRP$': {'desc': 'Possessive pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'RB': {'desc': 'Adverb (base)', 'cat': 'Adverb', 'color': 'tag-adv'},
    'RBR': {'desc': 'Adverb, comparative', 'cat': 'Adverb', 'color': 'tag-adv'},
    'RBS': {'desc': 'Adverb, superlative', 'cat': 'Adverb', 'color': 'tag-adv'},
    'RP': {'desc': 'Particle', 'cat': 'Other', 'color': 'tag-other'},
    'SYM': {'desc': 'Symbol', 'cat': 'Other', 'color': 'tag-other'},
    'TO': {'desc': "to' marker (infinitive/preposition)", 'cat': 'Preposition', 'color': 'tag-prep'},
    'UH': {'desc': 'Interjection', 'cat': 'Other', 'color': 'tag-other'},
    'VB': {'desc': 'Verb, base form (infinitive)', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBD': {'desc': 'Verb, past tense', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBG': {'desc': 'Verb, gerund or present participle', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBN': {'desc': 'Verb, past participle', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBP': {'desc': 'Verb, non-3rd person singular present', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBZ': {'desc': 'Verb, 3rd person singular present', 'cat': 'Verb', 'color': 'tag-verb'},
    'WDT': {'desc': 'Wh-determiner', 'cat': 'Determiner', 'color': 'tag-det'},
    'WP': {'desc': 'Wh-pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'WP$': {'desc': 'Possessive wh-pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'WRB': {'desc': 'Wh-adverb', 'cat': 'Adverb', 'color': 'tag-adv'},
    '.': {'desc': 'Sentence-final punctuation', 'cat': 'Punctuation', 'color': 'tag-punct'},
    ',': {'desc': 'Comma', 'cat': 'Punctuation', 'color': 'tag-punct'},
    ':': {'desc': 'Colon or ellipsis', 'cat': 'Punctuation', 'color': 'tag-punct'},
    '$': {'desc': 'Currency sign', 'cat': 'Symbol', 'color': 'tag-other'},
    '#': {'desc': 'Pound sign / hashtag', 'cat': 'Symbol', 'color': 'tag-other'},
    "''": {'desc': 'Closing quotation mark', 'cat': 'Punctuation', 'color': 'tag-punct'},
    '``': {'desc': 'Opening quotation mark', 'cat': 'Punctuation', 'color': 'tag-punct'},
    '(': {'desc': 'Opening parenthesis/bracket', 'cat': 'Punctuation', 'color': 'tag-punct'},
    ')': {'desc': 'Closing parenthesis/bracket', 'cat': 'Punctuation', 'color': 'tag-punct'}
}


def get_pos_explanation(tag: str) -> dict:
    """Returns human-friendly explanation, general category, and UI styling badge for a POS tag."""
    return POS_TAG_MAP.get(tag, {
        'desc': f'Part-of-speech tag ({tag})',
        'cat': 'Other',
        'color': 'tag-other'
    })


def penn_to_wordnet_pos(penn_tag: str):
    """Maps Penn Treebank POS tag to WordNet POS tag for lemmatization."""
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_word(word: str, penn_tag: str) -> str:
    """Lemmatizes a single word using WordNet based on its POS tag."""
    wn_tag = penn_to_wordnet_pos(penn_tag)
    try:
        lemma = lemmatizer.lemmatize(word.lower(), pos=wn_tag)
        # Restore case if original was capitalized
        if word.istitle():
            return lemma.capitalize()
        return lemma
    except Exception:
        return word.lower()


def analyze_text(text: str) -> dict:
    """
    Performs comprehensive NLP linguistic analysis on the provided text.
    
    Tasks performed:
    1. Sentence Tokenization
    2. Word Tokenization
    3. Part-of-Speech Tagging
    4. Lemmatization
    5. Structural & Lexical Metrics
    
    Returns:
        dict: Complete NLP linguistic breakdown
    """
    if not text or not text.strip():
        return {
            'sentences': [],
            'sentence_count': 0,
            'word_count': 0,
            'token_count': 0,
            'unique_words': 0,
            'lexical_diversity': 0.0,
            'avg_sentence_length': 0.0,
            'tokens': []
        }

    raw_text = text.strip()

    # Step 1: Sentence Tokenization
    try:
        sentences = sent_tokenize(raw_text)
    except Exception:
        # Fallback regex segmentation
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    # Step 2 & 3: Word Tokenization and POS Tagging
    try:
        tokens_raw = word_tokenize(raw_text)
    except Exception:
        tokens_raw = raw_text.split()

    try:
        pos_tags = pos_tag(tokens_raw)
    except Exception:
        pos_tags = [(t, 'NN') for t in tokens_raw]

    # Step 4: Token Detail Construction & Lemmatization
    analyzed_tokens = []
    pure_words = []
    lemmas_list = []

    for idx, (token, tag) in enumerate(pos_tags):
        is_punct = token in string.punctuation or tag in ['.', ',', ':', "''", '``', '(', ')']
        is_alpha = token.isalpha()
        tag_info = get_pos_explanation(tag)
        lemma = lemmatize_word(token, tag) if is_alpha else token

        if is_alpha:
            pure_words.append(token.lower())
            lemmas_list.append(lemma.lower())

        analyzed_tokens.append({
            'index': idx + 1,
            'token': token,
            'tag': tag,
            'category': tag_info['cat'],
            'description': tag_info['desc'],
            'color_class': tag_info['color'],
            'lemma': lemma,
            'is_punctuation': is_punct,
            'is_alpha': is_alpha
        })

    # Step 5: Syntactic and Lexical Metrics
    total_words = len(pure_words)
    unique_words = len(set(pure_words))
    lexical_diversity = round((unique_words / total_words) * 100, 1) if total_words > 0 else 0.0
    sentence_count = len(sentences) if sentences else 1
    avg_sentence_length = round(total_words / sentence_count, 1) if sentence_count > 0 else 0.0

    return {
        'sentence_count': len(sentences),
        'sentences': sentences,
        'word_count': total_words,
        'token_count': len(analyzed_tokens),
        'unique_words': unique_words,
        'lexical_diversity': lexical_diversity,
        'avg_sentence_length': avg_sentence_length,
        'tokens': analyzed_tokens
    }
