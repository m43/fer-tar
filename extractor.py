import math
import re
from abc import abstractmethod, ABC

import gensim
import numpy as np

import dataset
import sent2vec
import torch

PUNCT = '.!?'
RE_PUNCT = r'[?.!]'
WORD_VEC_PATH = "./Word2Vec/GoogleNews-vectors-negative300.bin"
SENT_VEC_PATH = "sentences/wiki_unigrams.bin"


class FeatureExtractor(ABC):
    """
    Abstract class to model features extractors
    """
    @abstractmethod
    def pre(self, x, **kwargs):
        """
        Extracts features from raw text, before tokenization.
        :param x: list of raw essays; list[str]
        :return: a tensor with extracted features; torch.tensor
        """
        pass

    def post(self, x, **kwargs):
        """
        Extracts features from preprocessed text, after tokeniyation.
        :param x: list of lists of tokens; list[list[str]]
        :return: a tensor with extracted features; torch.tensor
        """
        pass


class DummyExtractor(FeatureExtractor):
    """
    Dummy feature extractor that returns random vectors of given dimensionality
    """
    def __init__(self, **kwargs):
        self.dim = 100 if 'dim' not in kwargs.keys() else kwargs['dim']

    def pre(self, x, **kwargs):
        return torch.randn((len(x), self.dim))

    def post(self, x, **kwargs):
        return None


class BOWExtractor(FeatureExtractor):
    """
    Extracts bag-of-words features from the input tokens.
    """
    def __init__(self, **kwargs):
        from sklearn.feature_extraction.text import CountVectorizer
        _, x = kwargs['train_x']
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit([' '.join(xi) for xi in x])

    def pre(self, x, **kwargs):
        return None

    def post(self, x, **kwargs):
        coo = self.vectorizer.transform([' '.join(xi) for xi in x]).tocoo()

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


class W2VExtractor(FeatureExtractor):
    """
    Retrieves the word2vec vector representation of each word in the essay
    and averages them into a single vector.
    """
    def __init__(self, **kwargs):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(WORD_VEC_PATH, binary=True)
        self.mean = None
        self.stddev = None

    def pre(self, x, **kwargs):
        return None

    def post(self, x, **kwargs):
        vecs = []
        for tokens in x:
            vec = torch.empty((len(tokens), 300))
            for i, word in enumerate(tokens):
                try:
                    vec[i] = torch.tensor(self.model[word])
                except KeyError:  # out of vocabulary word
                    continue
            vecs.append(torch.mean(vec, dim=0))
        stack = torch.stack(vecs)
        if self.mean is None and self.stddev is None:
            self.mean = torch.mean(stack, dim=0)
            self.stddev = torch.sqrt(torch.var(stack, dim=0))
        return (stack - self.mean) / self.stddev


class S2VExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(SENT_VEC_PATH)

    def extract(self, x):
        vecs = []
        for i, text in enumerate(x):
            sentences = re.split(RE_PUNCT, text[1])
            embeddings = [torch.tensor(self.model.embed_sentence(s)) for s in sentences if s]
            embeddings = torch.stack(embeddings)
            vecs.append(torch.mean(embeddings, dim=0))
        return torch.stack(vecs)


class InterpunctionExtractor(FeatureExtractor):
    """
    Extracts interpunction counts for each input entry.
    """
    def __init__(self, **kwargs):
        pass

    def pre(self, x, **kwargs):
        return None

    def post(self, x, **kwargs):
        sc = kwargs['sc']
        if len(sc) == 0:
            return None
        counts_per_person = []
        for tokens in x:
            punct_count = torch.zeros(len(PUNCT))
            filtered = [t for t in tokens if t in PUNCT]
            for token in filtered:
                punct_count[PUNCT.find(token)] += 1
            counts_per_person.append(punct_count)
        stack = torch.stack(counts_per_person)
        for i in range(len(stack)):
            stack[i] /= sc[i]
        return stack


class CapitalizationExtractor(FeatureExtractor):
    """
    Extracts capitalization counts relative to number of sentences for each input entry.
    """
    def __init__(self, **kwargs):
        pass

    def pre(self, x, **kwargs):
        cap_per_person = torch.zeros((len(x), 1), dtype=torch.float32)
        sc = kwargs['sc']
        for i, entry in enumerate(x):
            cap_count = len([c for c in entry if c.isupper()])
            cap_per_person[i] = cap_count if sc[i] == 0 else cap_count / sc[i]
        return cap_per_person

    def post(self, x, **kwargs):
        return None


class RepeatingLettersExtractor(FeatureExtractor):
    """
    Extracts number of letters such that they form a sequence of more than 2 repeated letters.
    For example, in "He's just a toddlerrr" the number would be 3 (3 'r's in toddler).
    """
    def __init__(self, **kwargs):
        pass

    def pre(self, x, **kwargs):
        return None

    def post(self, x, **kwargs):
        values = []
        for tokens in x:
            values.append([0])
            count_reps = 1
            for t in tokens:
                for i in range(1, len(t)):
                    if (not t[i].isalpha()) or (t[i] != t[i - 1]):
                        if count_reps > 2:
                            values[-1][0] += count_reps
                        count_reps = 1
                    else:
                        count_reps += 1
        return torch.tensor(values, dtype=torch.float32)


class WordCountExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        _, x = kwargs['train_x']
        wc = torch.tensor([[1. * len([t for t in tokens if t not in PUNCT])] for tokens in x])
        self.mean = torch.mean(wc)
        self.stddev = torch.sqrt(torch.var(wc))

    def pre(self, x, **kwargs):
        return None

    def post(self, x, **kwargs):
        """
        Returns the number of tokens in each person's essay.
        """
        wc =  torch.tensor([[1. * len([t for t in tokens if t not in PUNCT])] for tokens in x])
        return (wc - self.mean) / self.stddev


if __name__ == '__main__':
    raw_x = ["Well, right now I just woke up from a mid-day nap. It's sort of weird, but ever since I moved to Texas, I have had problems concentrating on things. I remember starting my homework in  10th grade as soon as the clock struck 4 and not stopping until it was done. Of course it was easier, but I still did it. But when I moved here, the homework got a little more challenging and there was a lot more busy work, and so I decided not to spend hours doing it, and just getting by. But the thing was that I always paid attention in class and just plain out knew the stuff, and now that I look back, if I had really worked hard and stayed on track the last two years without getting  lazy, I would have been a genius, but hey, that's all good. It's too late to correct the past, but I don't really know how to stay focused n the future. The one thing I know is that when  people say that b/c they live on campus they can't concentrate, it's b. s. For me it would be easier there, but alas, I'm living at home under the watchful eye of my parents and a little nagging sister that just nags and nags and nags. You get my point. Another thing is, is that it's just a hassle to have to go all the way back to  school to just to go to library to study. I need to move out, but I don't know how to tell them. Don't get me wrong, I see where they're coming from and why they don't  want me to move out, but I need to get away and be on my own. They've sheltered me so much and I don't have a worry in the world. The only thing that they ask me to do is keep my room clean and help out with the business once in a while, but I can't even do that. But I need to. But I got enough money from UT to live at a dorm or apartment  next semester and I think I�ll take advantage of that. But off that topic now, I went to sixth street last night and had a blast. I haven't been there in so long. Now I know why I love Austin so much. When I lived in VA, I used to go up to DC all the time and had a blast, but here, there are so many students running around at night. I just want to have some fun and I know that I am responsible enough to be able to  have fun, but keep my priorities straight. Living at home, I can't go out at all without them asking where? with who?  why?  when are you coming back?  and all those  questions. I just wish I could be treated like a responsible person for once, but  my sister screwed that up for me. She went crazy the second she moved into college and messed up her whole college career by partying too much. And that's the ultimate reason that they don't want me to go and have fun. But I'm not little anymore,  and they need to let me go and explore the world, but I�m Indian; with Indian culture, with Indian values. They go against ""having fun. ""  I mean in the sense of meeting people or going out with people or partying or just plain having fun. My school is difficult already, but somehow I think that having more freedom will put more pressure on me to  do better in school b/c that's what my parents and ultimately I expect of myself. Well it's been fun writing, I don't know if you go anything out of this writing, but it helped me get some of my thoughts into order. So I hope you had fun reading it and good luck TA's.",
               "Well, here we go with the stream of consciousness essay. I used to do things like this in high school sometimes. They were pretty interesting, but I often find myself with a lack of things to say. I normally consider myself someone who gets straight to the point. I wonder if I should hit enter any time to send this back to the front. Maybe I'll fix it later. My friend is playing guitar in my room now. Sort of playing anyway. More like messing with it. He's still learning. There's a drawing on the wall next to me. Comic book characters I think, but I'm not sure who they are. It's been a while since I've kept up with comic's. I just heard a sound from ICQ. That's a chat program on the internet. I don't know too much about it so I can't really explain too well. Anyway, I hope I'm done with this by the time another friend comes over. It will be nice to talk to her again. She went home this weekend for Labor Day. So did my brother. I didn't go. I'm not sure why. No reason to go, I guess. Hmm. when did I start this. Wow, that was a long line. I guess I won't change it later. Okay, I'm running out of things to talk about. I've found that happens to me a lot in conversation. Not a very interesting person, I guess. Well, I don't know. It's something I'm working on. I'm in a class now that might help. The phone just rang. Should I get it?  The guy playing the guitar answered it for me. It's for my roommate. My suitemate just came in and started reading this. I'm uncomfortable with that. He's in the bathroom now. You know, this is a really boring piece of literature. I never realized how dull most everyday thoughts are. Then again, when you keep your mind constantly moving like this, there isn't really time to stop and think deeply about things. I wonder how long this is going to be. I think it's been about ten minutes now. Only my second line. How sad. Well, not really considering how long these lines are. Anyway, I wonder what I'm going to do the rest of the night. I guess there's always homework to do. I guess we'll see. This seat is uncomfortable. My back sort of hurts. I think I'm going to have arthritis when I get older. I always thought that I wouldn't like to grow old. Not too old, I suppose. I've always been a very active person. I have a fear of growing old, I think. I guess it'll go away as I age gradually. I don't know how well I'd deal with paralysis from an accident though. As long as I have God and my friends around, I'll be okay though. I'm pretty thirsty right now. There isn't much to drink around my room. Ultimate Frisbee, I haven't played that all summer. Fun game, but tiring. I'm out of shape. I'd like to get in better shape, but I hate running. It's too dull for me. Hmmm. it's almost over now. Just a few more minutes. Let's see if I make it to the next line. Short reachable goals!  Whatever. Anyway, what else do I have to do tonight. I guess I could read some. My shirt smells like dinner. It's pretty disgusting. I need to wake up for a 9:30 am class tomorrow. I remember when that wasn't early at all. Well, I made it to the next line. I'm so proud of myself. That's sarcasm, by the way. I wonder if I was suppose to right this thing as a narrative. Oh well too late now. Time for me to head out. Until next time, good bye and good luck. I don't know.",
               "An open keyboard and buttons to push. The thing finally worked and I need not use periods, commas and all those thinks. Double space after a period. We can't help it. I put spaces between my words and I do my happy little assignment of jibber-jabber. Babble babble babble for 20 relaxing minutes and I feel silly and grammatically incorrect. I am linked to an unknown reader. A graduate student with an absurd job. I type. I jabber and I think about dinoflagellates. About sunflower crosses and about the fiberglass that has be added to my lips via clove cigarettes and I think about things that I shouldn't be thinking. I know I shouldn't be thinking. or writing let's say/  So I don't. Thoughts don't solidify. They lodge in the back. behind my tongue maybe. Somewhere at the point of hiding but   dinoflaghelates, protistas and what was that sea weed. I think about the San Luiz valley and I think about the mushrooms in cow shit. I think about the ticos and I think about the chiggers that are living in my legs. I itch. I coat myself with clear nail polish in hopes to suffocate the bugs that are living in my legs and I remember Marco. I remember Ecuador  and I think about my thoughts and what I am not supposed to be doing in this assignment. Thoughts. I wonder if I think in sentences I wonder what affect my slowish typing has on my stream of consciousness and I wonder if there is a way that typing speed can be measured in this study  so that so link some generalization of dorky 301 psyc students. green and the table in my kitchen makes me want to vomit. orange. What an absurd color. wish I wasn't in the united state. My greencard runs out in a few years wonder what I do. I hope Dr. Linder gets back in his lab because I really need to find out if he has funds to pay me. May have to go back to the library. Brainless job of nothingness that would make me wallow in the world of boredom which isn't entirely bad. Need to focus on school organics and such. Period. Two spaces after the period. Mistakes and I want to eat not hungry and I wonder how many people talk about food in there little computer ramblings  Feel open and Happy that I am not having to edit this. Type type I don't know what I am think Hannah Imi and Osdprey house. I remember when I went down to that . she had spiders on hurt wall pain all over the place and we painted clouds on the ceiling and the blue walls were so obnoxious. Carey. Sex sex sex. yeah. This is a strange assignment and Portonoy's complaint is ringing in my head. Eager to finish so that I can start for Whom the Bell Tolls and get on with it. Bio and Carbon atoms bonds and orbitals. Thinking  about the electron configuration that surrounds the last letter in my first name and I think that I must have been granted a full ""s"" orbital  one up and one down. spinning on opposite directions and I am thinking about Scottish poetry about Mike in his kilt and about my guitar that I am slowly slowly slowly learning to play. I wonder what goes on in this study. I wonder if those happy little bored entertained grad students will scan words and I wonder how I can mess up this study? Random words like . don't know. ;Me me me me me and I wish that some things were easier and I wish that I had been keeping my eye on the clock. Wondering how long I have been typing and wishing that I was finished because I need to find out if I have to / will work in the Botany lab again and all that . ILS Belly and the Flamenco. Bjork and Rozamond Cockrill kickin' it in Saratoga Springs. I hate Molly's cat and wish that it could be exchanged for a worthwhile ferret. Type type type. I have managed to waste over 20 minutes of time I think. Who knows. What If I was to write this out and it took 30 minutes to write and 15 minutes to type. Thinking about nothing and wishing that some financial aid would come my way. Need a job and a sprinkling of time. Time to go and sign outta here. trees",
               "I can't believe it!  It's really happening!  My pulse is racing like mad. So this is what it's like. now I finally know what it feels like. just a few more steps. I wonder if he is going to get any sleep tonight!?  I sure won't!  Well, of course I have a million deadlines to meet tomorrow so I'll be up late anyway. But OH! I'm so so excited!  Yes!  Yes!  I can't believe it is finally happening. Wait! Calm down. We aren't officially a couple yet. What if I end up not liking him?  That would be horrible. Oh great, I wonder how long it'll take me to finish those Calculus problems?  I'll get it done. Don't you always, Amy?  I can't believe Bob did it!  He really did it!  He is THE miracle worker. If things turn out all right I will owe him more than I can ever repay. I wonder what Steve is doing in Malaysia right now?  An entire month!  I'll likely clean out his refrigerator by then. Omigosh!  Food, lunch tomorrow, what will I ever say to him?  He is perfect in every way imaginable. It is so important for him to think of me the same way. well, maybe not Perfect, but certainly dynamic. Who would have ever thought!  Good things do indeed come to those who wait!  Oh, I'll have to remember to sign the poster he made tomorrow morning. I hope Steve's alarm clock is reliable and I don't oversleep. That would be tragic if I slept 'til noon and missed the lunch. Thank goodness Portia is coming along. I will definitely need her support as well as Bob's. just having her there will take away some of the tension and put me more at ease. I'll have to rehearse what I say beforehand. things can only get better from here, right? hopefully. oh, I'm so nervous!  He will be too. maybe even more so. it'll be ok. Why in the world do humans put themselves through such torture. maybe love is really worth it?"]
    y = torch.tensor([[0.,1.,1.,0.,1.],
                      [0.,0.,1.,0.,0.],
                      [1.,0.,1.,1.,0.],
                      [1.,0.,1.,0.,1.]])

    raw_test_x = ["Well, here I go with the good old stream of consciousness assignment again. I feel like I'm back in freshman HS English class again. Not that that's a bad thing, mind you, but my English teacher freshman year made us do these assignments constantly, and mine were always completely ridiculous, like, ""wow, I'm really hungry. I wish I could go to Taco Bell. ""  They really had no point, except as busy work. In a psychology class, though, I can see the reasoning behind an assignment like this. Just letting my mind go free, and putting my random thoughts down in writing could be a big help in figuring out why I'm such a psychological screw-up. Well, that's not true. I don't want y'all getting the wrong idea about me, being that today was the first day of class and all. I'm really not a nut case. People may think I am, but really, I'm a normal kind of gal. Actually, down here in Texas, I guess I'm not normal. I don't like to eat biscuits and gravy for breakfast, and country fried steak with fried okra for dinner. I'm from Connecticut, and we don't even HAVE okra, much less worship it like it's some kind of vegetable goddess. My mind is starting to go blank--performance pressure I guess. I'm on the spot here--I don't want you all to be bored while you're reading this, if you ever do get around to reading this, that is. Well, I'm not going to stress just yet, so you're probably going to have to listen to some of my random, incoherent babbling for a few paragraphs. These computers are a big old pain in the ass. Here in the SMF, sure, they've got a bajillion computers, but unfortunately, we've got 42 bajillion students trying to use them, all at the same time. I think I'll be spending quite a few late, late nights in the computer center, just to get my stuff done. Yippee. That's what college is all about--late nights in the libraries. Yeah. Right. At this point, I don't even know what college is all about. I probably shouldn't say anything though, seeing as how I'm going to have to write another one of these thingys in a few days, where the topic is ""college""  Blah, blah, blah. I can't believe I'm actually doing this assignment on the same day that is was assigned!  Go me!  Talk about dedication. I really can't believe this. In high school, procrastination was my middle name. No, it was my first name. By second semester, I have more free periods in a day than actual classes, so I didn't have to do a damn thing. It was great!  Unfortunately, because of that, I'm going to have to work that much harder here at UT, to get those studying skills back up to par. High School. Now that was a trip. When I was there, I couldn't wait to get out of there. I hated that school, that town, everything except my friends, of course. Then, my family moved, right after graduation, and I learned real quick that there were worse places to be than in my old town. At least back home I have my friends and my boyfriend and my piece of crap car, and I knew what there was to do. After I moved, I had no friends, no life, no car, no nothing. I worked all day. That's it. now, though, I'm ready for this whole college thing. Austin seems like a fun city, where I might actually enjoy spending the next four years. Oh yeah. While we're on the subject of ""four years,""  why is it that all the professors & administrators that give speeches and stuff always make it sound like we'll be in college for like, 5 or 6 years?  I'm sorry, but I plan on graduating in 4 years. What's the problem here?  What are people doing, that they can't graduate in 4 years?  I just don't get it. no offense if any of y'all reading this took like 7 years to do your undergrad work. I'm not trying to knock you, just trying to figure this out. Well, it's 9:19, exactly 18 minutes after I started this nifty little piece of writing that makes no sense and has no point. I'm not really sure if I have fulfilled this assignment, like if I was supposed to analyze my personal stream of consciousness, where it took me, and what that means regarding my own personality. I guess if I had to, I could say that my mind works in mysterious ways, and even if the above essay seems to be illogically connected, to me, I can see the patterns. Yeah. I just went back & tried to read this over again, and I've got to give a suggestion. For these assignments, make the box we're writing in a box where you can see the whole line of writing at one time, without have to scroll across , because it's a real big pain for me, and I'm sure it's just as big a pain for you when you're trying to read it. Unless, of course, when you read it, you can see the whole line at once. I don't know, just a suggestion!  Thanks for taking the time to give us all the opportunity to get an easy 10 % of out final grade through these writing assignments!"]

    ext_hooks = (InterpunctionExtractor,  # W2VExtractor,
                 CapitalizationExtractor, RepeatingLettersExtractor,
                 WordCountExtractor)

    (trnx, trny), (valx, valy), (tesx, tesy) = dataset.load_features(ext_hooks, raw_x, y, 0.25, 0.25)
