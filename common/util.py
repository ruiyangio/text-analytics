import re

class Util(object):
    @staticmethod
    def normalizeString(text):
        #EMOJIS
        text = re.sub(r":\)", "emojihappy1", text)
        text = re.sub(r":P", "emojihappy2", text)
        text = re.sub(r":p", "emojihappy3", text)
        text = re.sub(r":>", "emojihappy4", text)
        text = re.sub(r":3", "emojihappy5", text)
        text = re.sub(r":D", "emojihappy6", text)
        text = re.sub(r" XD ", "emojihappy7", text)
        text = re.sub(r" <3 ", "emojihappy8", text)

        text = re.sub(r":\(", "emojisad9", text)
        text = re.sub(r":<", "emojisad10", text)
        text = re.sub(r":<", "emojisad11", text)
        text = re.sub(r">:\(", "emojisad12", text)

        #MENTIONS "(@)\w+"
        text = re.sub(r"(@)\w+", "mentiontoken", text)

        #WEBSITES
        text = re.sub(r"http(s)*:(\S)*", "linktoken", text)

        #STRANGE UNICODE \x...
        text = re.sub(r"\\x(\S)*", "", text)

        #General Cleanup and Symbols
        text = re.sub(r"[^A-Za-z0-9(),:.!;?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        # text = re.sub(r",", " , ", text)
        # text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip().lower()

    @staticmethod
    def defaultTokenize(text):
        text = Util.normalizeString(text)
        # return re.split(r"\s+", text)
        return Util.negate_sequence(text)

    @staticmethod
    def negate_sequence(text):
        negation = False
        delims = "?.,!:;"
        result = []
        words = text.split()
        prev = None
        pprev = None
        for word in words:
            stripped = word.strip(delims).lower()
            if not stripped:
                continue
            negated = "not_" + stripped if negation else stripped
            result.append(negated)
            if prev:
                bigram = prev + " " + negated
                result.append(bigram)
                if pprev:
                    trigram = pprev + " " + bigram
                    result.append(trigram)
                pprev = prev
            prev = negated

            if any(neg in word for neg in ["not", "n't", "no"]):
                negation = not negation

            if any(c in word for c in delims):
                negation = False

        return result