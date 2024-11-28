"""
Sentiment Classification using BERT
"""

import os
import re
import warnings
from collections import Counter

import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification


warnings.filterwarnings("ignore")


LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

GEN_LABELS = {
    0: "negative",
    1: "positive"
}

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CACHE_DIR = os.path.expanduser("~/cls-cond-mdlm/db/modules")

BLOCK_SIZE = 512
BATCH_SIZE = 16

# Preprocessing regexes

PUNC = re.compile(r"[.?!]+")
QUOTES_OR_NEWLINE = re.compile(r"[\"\n]")
SPACES = re.compile(r"\s+") # replace arbitrary number of spaces with a single space



def split(text:str)->list[str]:
    def _preprocess(text: str) -> str:
        text = text.strip()
        text = QUOTES_OR_NEWLINE.sub("", text) # remove quotes and newlines
        text = SPACES.sub(" ", text) # replace arbitrary number of spaces with a single space
        return text

    beginings = re.split(PUNC, text)
    puncs = re.findall(PUNC, text) + [""] # match length of beginings
    parts = [begining + punc for begining, punc in zip(beginings, puncs)]
    parts = [_preprocess(part) for part in parts]
    # Remove leading/trailing whitespaces and empty strings
    parts = [part.strip() for part in parts if part.strip()]
    return parts


def infer(texts: list[str],
          tokenizer: AutoTokenizer,
          model: RobertaForSequenceClassification,
          aggregate="mean",
          verbose=0,
          ) -> list[int]:
    """
    Perform inference on a list of texts using a pre-trained RoBERTa model for sequence classification.
    Args:
        texts (list[str]): A list of input texts to classify.
        tokenizer (AutoTokenizer): The tokenizer to preprocess the input texts.
        model (RobertaForSequenceClassification): The pre-trained RoBERTa model for sequence classification.
        aggregate (str, optional): The method to aggregate predictions across chunks.
                                    Options are "mean" or "max_count". Defaults to "mean".
    Returns:
        list[int]: A list of predicted labels for each input text.
    """

    def _infer_chunk_batched(chunked_texts: list[str]) -> list[int]:
        # rebatch at the beginning of each chunk
        logits = torch.zeros(len(chunked_texts), 3)
        for i in range(0, len(chunked_texts), BATCH_SIZE):
            batch = chunked_texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch,
                               padding=True,
                               truncation=True,
                               return_tensors="pt",
                               max_length=BLOCK_SIZE).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits[i:i+len(batch)] = outputs.logits
        if verbose:
            print("Mean logits:")
            print(torch.mean(logits, dim=0))
            print("Majority vote:")
            c = Counter(logits.argmax(dim=1).tolist())
            print(c)
        if verbose > 1:
            print(logits)
            print(logits)
        # remove logits for neutral class
        logits = logits[:, [0, 2]]
        if aggregate == "mean":
            pred_label = torch.mean(logits, dim=0).argmax().item()
        elif aggregate == "max_count":
            pred_label = torch.mode(logits.argmax(dim=1)).values.item()
        else:
            raise ValueError(f"Invalid value for aggregate: {aggregate}")
        return pred_label

    preds = []
    for text in texts:
        chunks = split(text)

        pred = _infer_chunk_batched(chunks)
        preds.append(pred)

    return preds

def get_models(device) -> tuple[AutoTokenizer, RobertaForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL, cache_dir=CACHE_DIR).to(device)
    return tokenizer, model

def test_infer(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = get_models(device)

    preds = infer(texts, tokenizer, model)
    for text, pred in zip(texts, preds):
        print(f"Text: {text}")
        print(f"Predicted sentiment: {LABELS[pred]}")

class SentimentClassifier:
    def __init__(self, aggregate="max_count", verbose=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = get_models(self.device)
        self.aggregate = aggregate
        self.verbose = verbose

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def predict(self, texts: list[str], force_agg=None) -> list[str]:
        if force_agg is not None:
            agg = force_agg
        else:
            agg = self.aggregate

        preds = infer(texts,
                      self.tokenizer,
                      self.model,
                      aggregate=agg,
                      verbose=self.verbose)

        return preds

    def get_class(self, label: int) -> str:
        return LABELS.get(label, "unknown")

    def compute_accuracy(self, gen_labels: list[int], gt_labels: list[int]) -> float:
        correct = 0
        for gen_label, gt_label in zip(gen_labels, gt_labels):
            if LABELS[gen_label] == LABELS[gt_label]: # different labels for gen and gt
                correct += 1
        return correct / len(gt_labels)

if __name__ == "__main__":

    # dummy text
    test_text1 = """
    don\'t know it hear me, but I would rather listen to it and NECK IT OFF:BUT! Finally, this CD is undionable, it is still a pretty solid release and I reccomend it to anyone who doesn\'t have it.Talk about Marittis\' A powerful romp out of Fr. Harris. A cutsy girlfriend and Lady Joe develop real people and parents.Strong harmony and extraordinary voices together. A first CD that is an empowering place to start with.The Wotheest Music Ever! Anyone on and off can claim that there is no way of Irish folks on this Worship album!I found this CD to rank as the Greatest in 1953 bands!Taking an important, defined characters, deep/mudded downs, affairs and frivolous changes you never heard on an Irish CD.FURDER I bought the Sky Pe 1 and then put it in my car a lost place that was pouring out of my eyes...and I played it to get more and still in my car... Feeling so damn sad... and I love the song in my loop..Excellent CD Cranas have been the old voice benitort on vinyl for a very long time now...a total waste of money I bought a lather several years ago on the Rations and she finally sold out JR after she was versed on her musical business in 1981. I was thrilled with her collection of her records, received her first album in 1987 and obtained it out of a way as I stumbled upon to produce an Art. Power Age music album. Finding voices that we can visualize the use of these songs, are in a way to be able to burn the taste of the listener. On top heavy, audience lyrics, interesting songs, added vocals that take themselves and really get their way. Definitely a relax when listening to CD, I will then leave that statement of why this songive could be so rich to become micrusions to my own. Its awsome music to be time.Best Blend I have had no problem with joja, but I taste and drink all my tired prehear rams, and the bulk of regular blueberries usually add up to the expense of drinking curry from the YugToba oil. It has high levels of sugar and vanilla that we never used before. It\'s unavailable and there\'ll still be a few brands I can just add to this model and work everytime I reheat my own beans. When I drink coffee and I drink this I slow down the bottle of Starbucks. I just pour me and can brew several servings/ quantities of milk. Now it never got too cold and keeps coming out.The Secret Easy to setup. I couldn\'t get anyone to test it yet - no burning buildup with very low water. Needless to say, I buy this again now, and no problems at all!Great tunes this album is a traditional tripir rock style. I realized some of these marvelous songs was unavailable, the single, "Foun of Garden" - "I\'ve Got Back to Land," finally carries the aforementioned "Hym v Angeles" to the breathtaking harmonies. "Dream Lord\'s Hand" is one of the best Sepophonic riddle tracks. I personally think other songs on "Hot Bang" are their best, but overall lacking a rather interesting and heartfelt love song and "A**e".typical, but modern sound. I am somewhat nurtured by what this style of recording was, but the reviews yet have caught a faint risk just to get a little annoyed. I find this quelan recording to be scarier than I can with that. Just entering my head a great record label. Try doing so solo, they\'ll be old things and melodies so they sit up and feast and thoroughly grow up. I recommend the similar tome in particular, "love you wife" for relaxing, relaxing listen to production. I can\'t wait for more selections.Enchanting! I thank you for Fern Catherine\'s writings! She soothes all Maria\'s large Irish women. They have topped the pen of Aphosphorian with religious needs and the share of the "all small elephants" and I loved Welo\'s writings, but one of the suave choices ever for experince and I Mayin won\'t touch it to everyone.WOW!!! What a great CD, girl! I\'m so happy with the CD. Well done, and the reasons I love, love, loving them. That just don\'t have to be beat here! I can only guess that they just prove that she "saikes", that everyone\'s! They end up with freaky stuff with some contrived captions. I like, like Pans, want a great fun song, but make a cat to who you are. Other CDs don\'t buy"? "A+"!It\'s nice to own a collection of all these!!! Because I sat in the streets lots, taped on the road.You should have it. You should absolutely have a bow for this CD it would separate.'
    """  # noqa: E501
    test_text2 = """
    I know it\'ll work, but I would rather get this system and set it up in Windows Vista. Finally, the CD is undependable, it is to impossible, put multiple CDs, rewind it as slowly, and it doesn\'t work even once. It was made for approximately a never buy as the scanner so you can. I wouldn\'t return it and if it didn\'t, I would. The worst and for junkies waste your money.Does NOT edit in twire burner This is my first GB disk... Ever..and it just stop working. Did not bother by saying alot of folks like this but no way!, right? Had to load out the USB port for the drive to be important, just don\'t buy it... Lasted over 2 months and purchased a regular DVD burner with an alternate photo.FURPIABLE PRODUCT This microdisk drive has some odd bugs but it only worked fine for mem 9 out of my turntable (maybe over 50 if I was still in my car). I tried to send data files I activate the item and my devices meant run but didn\'t work. So, I transferred a 10 disc hard drive for downloading software before getting released.Don\'t waste your money I bought a product purchased several years ago.amazon infuriations. The finally V rating is down. This product I will not use in my home. Obviously, they will not forthcoming my drive. I received something that is now dead and then it walks of run way to drive which apparently wont produce disk 21. it will freeze up. they recommend that it says that you buy only one disk, however they will not even be able to burn a copy of it. i do not recommend any buyer purchasing this product, despite the unfair posting titles themselves require free additional peoples tax. lesson learned.Karaoke CD machine I will buy this product and find why it was on sale but had this problem and I am tempted to buy a whole new one.It may be time again for a bad review with no luck. I hate the final tragedy of my purchase.Buyer beware! I bought this Machine in the course of breakup. We have had trouble finding the last number of discs from our Creative Tape collection to contact the manufacturer to buy a working one. One that we never listen to Apple computer and store software, and still everything else that just appears should be able to get it to work! There is no quality control or disappointment. Great customer service and suggestions forum, so I\'ll probably just give it to some other speakers to whom I can.Never Buys Pacific Digital Products I was in Texas and have had several experiences with them. I noticed it outputally because I couldn\'t get it but can\'t get any sound burning at a very low price. Needless to say do not buy this unit, thus seeing no warnings about the product used. Given the bargain price, I managed to connect it to my PC in a bookshelf scanner, once again, a time resounding search channels.Beware Not "hard to use," most of the CD tracks on the CD were faulty and got a message that it should of been pulled off its own software.Jitter Sepophonic Sound and Explo I ordered this product from their leapmasters. Most of their products aren\'t particularly poor and does not say I would find if "client" would give me any bass response or sound. I am not holding for help. My computer generated on my printer using the internet. Don\'t take the risk just to get a little annoyed with non-existent support. Way over to give thierd response to their products. I\'ll probably give them a try or another odd product so that, they\'ll be happy.Good Product Idea - Bad Experience Customer Service As others have pointed out of customer service as mentioned prior in purchase, before telling my wife that returning the "fan" it would be one product (one star), if it continued to that two weeks had expired.Amazon took multiple two-four requests to package it\'s large defective set.Not compatible with the N64 I bought two of them. One and the share of old TDK never worked. I connected them and uninstalled the package and noGB would not print.Needless to say I guess product won\'t do it to the PSWOW!!! Tell me you should charge for these before you purchase an ipod? DON\'T BUY THIS PRODUCT!! And, if you use them at all you don\'t have to be durable.Good cause After only two months they lasted 5 months! Don\'t buy this product that their website (Amazon) doesn\'t sell their stuff anymore.arrived was lots of chips to a formatted machine These two CDs work great, just did the same thing when I read the CDs. Both CDs have 5"x". At a time it\'s time to file a collection that doesn\'t interface at all like the drive at lots.Now these are expensive.You\'d seem to have a my copy of a DVD Creator I have never designed before.'
    """  # noqa: E501
    test_text3 = """
 wanted to know it hear it, but I would bet the rest of the set would go in your chest. Finally, this CD again....Excellent variety This book is very thorough, distilled into several details of recent tenure, but in way it doesn\'t have a single one about Marittis. A powerful, sharp out of mind biochemistry critic who covers everything in qualitative and scientific speculation. Beckman does a good job composing and explains many factors. She also does a good job explaining economics to start with.The Wirtest Book Ever I am 27 and you can tell that there\'s no better than the folks against this author. Sometimes it\'s hard for me to apply what the facts would be like referring to each important, what characters we all have mudded language, affairs and their changes... all together was an utter achievement. It was amazing to make words just that manageable.. I put this book down and have a place that was pouring out of my eyes...and I would have to get into an outlet in my car before the airport never shifts... and I have the best in my life...Excellent CD Marranas music. I\'ve been a Quorthe fan for a very long time. This is the beginning of a band of a lovely voice. He goes through some inspirations and then finally ceases to portrait a proseitely wonderful album on the late 19th century.Some of these songs are one of those Sanshe received her first album, Prince and sang it all of a way.Be latter chosen to produce something different.New Age music album. Finding voices that children can visualize the expression of these can be found in everyday life which leap to themselves and defies of changes. That Onven, mature audience group will surely listen to, and will take themselves away to additional journeys and jazz and a relax while listening to CD, which will often leave you smiling.Awesome Product I love the taste of this food from micrusions to the taste. Its awsome. I be time consuming for hours and have reached no loss or loss in the evening I drink and drink.Excellent Holghhear Performances I had the pleasure of hearing Altaram Obel of the last forty minute release from the years icon of the favorite composer of the club. This was a recording that we never played before. Released several decades since it was still recorded in LP, I\'ll be able to play this wonderful work again at the end of my own house.Wonderful voice and a beautiful piano! I never thought so much of this recording piano. I listened to it once and over again and again when I was in love with the version of Mary\'s Mass. As an acquaintance I love Francine and then get eternal excited. She goes on CD Callably with Mirre for a gift to give. Go buy this CD already! Both are fantastic concert recordings!Great tunes; but are absolutely traditional Christmas songs This is a very good compilation of these marvelous songs -- unavailable, the single, and "A Love Garden." My favorite song is "Excellent Sing Years," most of the reason this Hyma favorites are "Cortunate Mary" and "The Lord" and "Variah Hymice Sepophonic". Other tracks (Wholety Noel!) on "To Bang The Mission" (check for particularly "You\'re a Restaurant Culture Edition) and "Regiente Faithtypical Latin Christmas".Powerful I am not able to help create a level of education about one of the Moses yet I had the faint risk just to get a little annoyed. I bought this quiltan book to give husbands good examples to get their attention. This book on meditation also great for me. If I do that, they\'ll be able to entourage myself and see where I am learning to grow. The illustrations of the synacles by the author, "with my wife" has been one concern for it itself. I would recommend providing an evaluation on what is to do it. Whether you are really religious savvy or rate at beginners, then it\'s visually motivating and helpful as well as the result of target obligation in dealing with religious practices and culture.Excellent! I really like this book. This is soo more helpful, but very helpful and gives better condition to the experince upon in actions and practice. I recommend it highly.WOW!!! Very informative, informative, very informative! There was a very cute section. It was great and at a great price, love the book!Awesome That you don\'t have to be a geek, cause I only bought this CD and really wanted it to be. It is a fun soundtrack! Some songs are okay, but there are some contrived captions of clips to enjoy, so I guess it was great to buy! Also, a warning to people who are looking for CDs don\'t buy.Bad deal but time it\'s rare to find a collection here: Great fun! I worked in the streets lots of fun on the road.You don\'t think that you should absolutely have a DVD that I have anyone would need.
    """  # noqa: E501

    stm_cls = SentimentClassifier(verbose=1)

    test_text = [test_text1, test_text2, test_text3]

    print(stm_cls.predict(test_text, force_agg="mean"))
