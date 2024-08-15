'''
This code was taken and modified from the following repository: https://github.com/disooqi/ArabicProcessingCog
'''
import sys
sys.path.insert(0, "/onyx/data/p118/POST-THESIS/original_data/Arabic_Processing")
import re
import config
from script import *
from nltk.corpus import stopwords
stopwords_arabic = stopwords.words('arabic')


class ArabicNormalizer:

    def __init__(self):
        # self.StopWordRemover = StopwordRemover()
        self.norm_table = {
            ALEF_MADDA: ALEF,
            ALEF_HAMZA_ABOVE: ALEF,
            ALEF_HAMZA_BELOW: ALEF,

            TEH_MARBUTA: HEH,
            ALEF_MAKSURA: YEH,

            TATWEEL: u'',

            # Ligatures
            LAM_ALEF: LAM + ALEF,
            LAM_ALEF_HAMZA_ABOVE: LAM + ALEF,
            LAM_ALEF_HAMZA_BELOW: LAM + ALEF,
            LAM_ALEF_MADDA_ABOVE: LAM + ALEF,

            # Diacritics
            FATHATAN: u'', DAMMATAN: u'',
            KASRATAN: u'', FATHA: u'',
            DAMMA: u'', KASRA: u'',
            SHADDA: u'', SUKUN: u'',

            # Numbers English
            ZERO: u'',
            ONE: u'',
            TWO: u'',
            THREE: u'',
            FOUR: u'',
            FIVE: u'',
            SIX: u'',
            SEVEN: u'',
            EIGHT: u'',
            NINE: u'',

            # Numbers Arabic
            ar_ZERO: u'',
            ar_ONE: u'',
            ar_TWO: u'',
            ar_THREE: u'',
            ar_FOUR: u'',
            ar_FIVE: u'',
            ar_SIX: u'',
            ar_SEVEN: u'',
            ar_EIGHT: u'',
            ar_NINE: u''
        }

        # For normalizing sentences by removing punctuation marks,
        # We will not remove full stop marks because they will help
        # identify sentences, since Word2Vec need to be trained on sentences
        self.punctuation_norm_table = {
            ar_COMMA: u'',
            ar_SEMICOLON: u'',
            ar_QUESTION: u'',
            ar_PERCENT: u'',
            ar_DECIMAL: u'',
            ar_THOUSANDS: u'',
            # ar_FULL_STOP: u'', # commented out for in order to identify sentences
            EXCLAMATION: u'',
            en_QUOTATION: u'',
            NUMBER_SIGN: u'',
            DOLLAR_SIGN: u'',
            en_PERCENT: u'',
            AMPERSAND: u'',
            LEFT_PARENTHESIS: u'',
            RIGHT_PARENTHESIS: u'',
            ASTERISK: u'',
            PLUS_SIGN: u'',
            en_COMMA: u'',
            HYPHEN_MINUS: u'',
            # en_FULL_STOP: u'', # commented out for in order to identify sentences
            SLASH: u'',
            en_COLON: u'',
            en_SEMICOLON: u'',
            en_LESS_THAN: u'',
            en_EQUALS_SIGN: u'',
            en_GREATER_THAN: u'',
            en_QUESTION: u'',
            COMMERCIAL_AT: u'',
            LEFT_SQUARE_BRACKET: u'',
            BACKSLASH: u'',
            RIGHT_SQUARE_BRACKET: u'',
            CIRCUMFLEX_ACCENT: u'',
            UNDERSCORE: u'',
            GRAVE_ACCENT: u'',
            LEFT_CURLY_BRACKET: u'',
            VERTICAL_LINE: u'',
            RIGHT_CURLY_BRACKET: u'',
            TILDE: u'',
            Leftpointing_double_angle_quotation_mark: u'',
            MIDDLE_DOT: u'',
            Rightpointing_double_angle_quotation_mark: u'',
            COPYRIGHT: u'',
            RIGHT_TO_LEFT_MARK: u'',
            LEFT_TO_RIGHT_MARK: u''
        }

    def __str__(self):
        return 'trying'

    def __del__(self):
        pass

    def normalize_token(self, token):
        ''' a token could be a single word, a multiword expression, or a named entity '''
        # check if token is a valid Arabic Word (from Taha Zerrouki)
        if config.remove_stopwords:
            if token in stopwords_arabic:
                return ''
        if not isArabicword(token):
            if config.remove_nonArabic_word:
                return ''
            elif not config.process_nonArabic_word:
                return token

        # if config.tweet_normalization:
        #     token = re.sub(r'(.)\1+', r'\1\1', token)

        if config.remove_repeating_characters:
            token = re.sub(r'(.)\1+', r'\1', token)

        term = list()
        # loop over each character
        for char in token:
            if char == ALEF_HAMZA_ABOVE and not config.replace_ALEF_HAMZA_ABOVE:
                term.append(char)
            elif char == ALEF_HAMZA_BELOW and not config.replace_ALEF_HAMZA_BELOW:
                term.append(char)
            elif char == ALEF_MADDA and not config.replace_ALEF_MADDA:
                term.append(char)
            elif char == ALEF_MAKSURA and not config.replace_ALEF_MAKSURA:
                term.append(char)
            elif char == TEH_MARBUTA and not config.replace_TEH_MARBUTA:
                term.append(char)
            elif char == TATWEEL and not config.remove_KASHIDA:
                term.append(char)
            elif char in TASHKEEL and not config.remove_diacritics:
                term.append(char)
            else:
                # Note: dictionary.get allows you to provide an additional default value if the key is not found
                term.append(self.norm_table.get(char, char))

        return ''.join(term)

    def normalize_sentence(self, line):
        '''
            Normalize the sentence, if we have senetences generated from .readlines()
            :param line: the sentence
            :return: list of tokens normalized and concatenated into one string
        '''
        if config.Add_SPACE_after_TEH_MARBUTA:
            line = line.replace(TEH_MARBUTA, TEH_MARBUTA+SPACE)

        # for ch in PUNCTUATIONS:
        for ch in self.punctuation_norm_table:
            if config.remove_punc:
                line = line.replace(ch, SPACE)
                continue
                
            if config.isolate_punc:
                line = line.replace(ch, SPACE+ch+SPACE)    
            
            if config.replace_punc:
                # Note: dictionary.get allows you to provide an additional
                # default value if the key is not found
                tempch = self.punctuation_norm_table.get(ch, ch)
                line = line.replace(ch, tempch)
            
        tokens = line.strip().split()

        if config.ignore_oneword_line and len(tokens)==1:
            return ''
        
        terms = list()
        for token in tokens[:]:
            term = self.normalize_token(token)
            if term.strip():
                terms.append(term)

        return ' '.join(terms)


if __name__ == '__main__':

    arabnormalizer = ArabicNormalizer()

    # read the entire txt file, and normalze
    with open('82032809.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        cleaned_lines = []
        for line in lines:
            if line.strip() in ['\n', '']:
                continue
            else:
                c_line = arabnormalizer.normalize_sentence(line=line)
                cleaned_lines.append(c_line)
    f.close()

    with open('82032809_cleaned.txt', 'w', encoding='utf-8') as f:
        # writing delimiters like this results in a tuple of unicode delimiters
        delimiters = EXCLAMATION, en_FULL_STOP, en_SEMICOLON, en_QUESTION, ar_FULL_STOP, ar_SEMICOLON, ar_QUESTION

        # re.escape allows to build the pattern automatically and have the delimiters escaped nicely
        regexPattern = '|'.join(map(re.escape, delimiters))

        # read the text and split into sentences whenever a delimiter from delimiters above is encountered
        for line in cleaned_lines:
            sentences = re.split(regexPattern, line)

            for sent in sentences:
                if sent.strip() in ['', '\n']:
                    continue
                f.write(sent + '\n')
        f.close()


















    
