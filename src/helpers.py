import settings
import os
import pathlib
import nltk
from nltk.corpus import stopwords


def tokenize(text):
    return [token.lower() for token in nltk.word_tokenize(text) if any(char.isalpha() for char in token)]


# ------------------------------------------------------------------------
# functions words
# ------------------------------------------------------------------------

latin_function_words = ['et', 'in', 'de', 'ad', 'non', 'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi', 'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur',
                        'circa', 'quidem', 'supra', 'ante', 'adhuc', 'seu', 'apud', 'olim', 'statim', 'satis', 'ob',
                        'quoniam', 'postea', 'nunquam', 'semper', 'licet', 'uidelicet', 'quoque', 'uelut', 'quot']


# return list of function words
def get_function_words(lang):
    if lang == 'latin':
        return latin_function_words
    elif lang in ['english', 'spanish', 'italian']:
        return stopwords.words(lang)
    else:
        raise ValueError('{} not in scope!'.format(lang))



def tee(msg, log):
    print(msg)
    log.write(f'{msg}\n')
    log.flush()


def check_author(args):
    if args.positive == 'ALL':
        args.authors = list_authors(args.corpuspath, skip_prefix='Epistola')
    else:
        if (args.positive not in settings.AUTHORS_CORPUS_I) and (args.positive in settings.AUTHORS_CORPUS_II):
            print(f'warning: author {args.positive} is not in the known list of authors for CORPUS I nor CORPUS II')
        assert args.positive in list_authors(args.corpuspath, skip_prefix='Epistola'), 'unexpected author'
        args.authors = [args.positive]


def check_feat_sel_range(args):
    if not isinstance(args.featsel, float):
        if isinstance(args.featsel, str) and '.' in args.featsel:
            args.featsel = float(args.featsel)
        else:
            args.featsel = int(args.featsel)
    if isinstance(args.featsel, float):
        assert 0 < args.featsel <= 1, 'feature selection ratio out of range'


def check_class_weight(args):
    assert args.class_weight in ['balanced', 'none', 'None']
    if args.class_weight.lower() == 'none':
        args.class_weight = None


def check_corpus_path(args):
    assert os.path.exists(args.corpuspath), f'corpus path {args.corpuspath} does not exist'
    args.corpus_name = pathlib.Path(args.corpuspath).name


def check_learner(args):
    assert args.learner.lower() in settings.param_grid.keys(), \
        f'unknown learner, use any in {settings.param_grid.keys()}'


def check_log_loo(args):
    if args.log is None:
        os.makedirs('../results', exist_ok=True)
        args.log = f'../results/LOO_Corpus{args.corpus_name}.Author{args.positive}.' \
                   f'fs{args.featsel}.classweight{str(args.class_weight)}.CLS{args.learner}.txt'


def check_log_unknown(args):
    args.unknown_name = pathlib.Path(args.unknown).name
    if args.log is None:
        os.makedirs('../results', exist_ok=True)
        assert os.path.exists(args.unknown), f'file {args.unknown} does not exist'
        args.log = f'../results/Unknown{args.unknown_name}_Corpus{args.corpus_name}.Author{args.positive}.' \
                   f'fs{args.featsel}.classweight{str(args.class_weight)}.CLS{args.learner}.txt'