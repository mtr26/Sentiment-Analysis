import torch as th
from transformers import BertTokenizer
from model import Model

device = "cpu"#"mps" if th.backends.mps.is_available() else "cpu"

model = th.load("Model_Complex_trained", map_location=device)



model = model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# List of 50 positive sentences
positive_sentences = [
    "I am feeling happy today because the sun is shining.",
    "The new cafe on the corner offers delightful pastries and coffee.",
    "I love spending time with my family at the park.",
    "The project at work was successful and received positive feedback.",
    "I enjoyed the movie last night; it was heartwarming.",
    "My best friend surprised me with a thoughtful gift.",
    "The weather is perfect for a day at the beach.",
    "I had a productive morning at work.",
    "The delicious dinner left me feeling satisfied and happy.",
    "I am excited about my upcoming vacation.",
    "My favorite song always lifts my mood.",
    "I appreciate the kindness of strangers.",
    "The garden is blooming with vibrant flowers.",
    "I received a compliment that made my day brighter.",
    "I enjoyed a refreshing walk in the cool evening air.",
    "I had a delightful conversation with an old friend.",
    "The new book I read was both informative and inspiring.",
    "I feel grateful for all the love and support around me.",
    "The sunset painted the sky with beautiful colors.",
    "I am proud of my accomplishments and hard work.",
    "The lively music made the party fun and energetic.",
    "I am excited to start a new creative project.",
    "I found joy in simple moments like a warm cup of tea.",
    "The vibrant city lights made the night magical.",
    "I had a good laugh with my coworkers.",
    "I love the peaceful sound of the rain.",
    "I enjoyed the delicious dessert at the new bakery.",
    "I feel confident and optimistic about the future.",
    "The museum exhibit was both educational and enjoyable.",
    "I cherish every moment spent with loved ones.",
    "The calm ocean waves brought me tranquility.",
    "I am thrilled about the progress of my new hobby.",
    "I felt appreciated when someone thanked me for my help.",
    "I enjoyed the warm ambiance of the cozy café.",
    "I woke up feeling refreshed and full of energy.",
    "The artwork in the gallery was breathtaking.",
    "I had an amazing day exploring a new city.",
    "I appreciate the beauty of nature all around me.",
    "I am motivated to achieve my goals.",
    "I felt joyful when I reconnected with an old friend.",
    "I had a peaceful day meditating in the park.",
    "I love the way music fills my heart with happiness.",
    "I enjoyed a wonderful meal with great company.",
    "I feel inspired to try something new.",
    "The laughter of children brightened my day.",
    "I am excited about the creative ideas I have.",
    "I appreciate every small victory in life.",
    "I felt happy as I walked through the blooming garden.",
    "I am grateful for the sunny day and clear skies.",
    "I had a fantastic time at the community event.",
    "I woke up with a smile, feeling optimistic about the day ahead.",
    "The coffee shop had a welcoming atmosphere that boosted my mood.",
    "I received a warm compliment that made me feel appreciated.",
    "Spending time in nature always fills me with joy and peace.",
    "I celebrated my success with friends and felt truly proud.",
    "The kind gesture from a stranger brightened my afternoon.",
    "A surprise visit from an old friend filled me with happiness.",
    "I felt the love in the air during the community festival.",
    "Reading an uplifting book inspired me to pursue my dreams.",
    "I enjoyed a serene walk by the lake that calmed my mind.",
    "The lively conversation at the dinner table made me feel connected.",
    "I found delight in the beauty of a clear, starry night.",
    "The cheerful music at the event lifted everyone's spirits.",
    "I am grateful for the supportive team that motivates me every day.",
    "My creative ideas flowed effortlessly during the brainstorming session.",
    "I felt accomplished after finishing a challenging project.",
    "A fun day at the amusement park left me feeling exhilarated.",
    "I cherish the warm memories of a happy family reunion.",
    "The soft breeze on a sunny day made me feel alive and content.",
    "I enjoyed a delicious meal prepared by a talented chef.",
    "I felt truly blessed as I watched the sunrise this morning.",
    "I am filled with gratitude for the positive people in my life.",
    "The vibrant art display sparked joy and creativity in me.",
    "I enjoyed a relaxing afternoon in the peaceful garden.",
    "The enthusiastic applause at the concert made me feel valued.",
    "I am overjoyed by the unexpected success of my project.",
    "A heartfelt thank you from a friend made me smile.",
    "I feel at ease in the calming environment of my favorite park.",
    "I was uplifted by the motivational speech at the event.",
    "The festive decorations brightened the entire neighborhood.",
    "I felt encouraged by the kind words of my mentor.",
    "A day spent at the beach filled me with positive energy.",
    "I enjoyed the burst of colors in the fall season.",
    "I felt rejuvenated after a peaceful yoga session.",
    "I appreciate the small joys that make each day special.",
    "The lively market was filled with smiles and friendly faces.",
    "I am thankful for the opportunities that bring growth and learning.",
    "The heartwarming story shared by a friend lifted my spirits.",
    "I enjoyed the relaxing aroma of fresh flowers in the garden.",
    "A surprise gift made me feel incredibly valued and loved.",
    "I had a fun adventure exploring the hidden gems of the city.",
    "I am thankful for the gentle reminder to appreciate life's beauty.",
    "The calm ambiance of the library provided a peaceful retreat.",
    "I was delighted by the thoughtful gesture of my coworker.",
    "A sunny afternoon in the park brought a sense of peace.",
    "I felt energized by the creative workshop I attended.",
    "I appreciate every opportunity to learn something new.",
    "A spontaneous dance in the rain made me feel free.",
    "I am excited about the promising future that lies ahead.",
    "I felt a deep sense of satisfaction after helping a neighbor."
]

# List of 50 negative sentences
negative_sentences = [
    "I feel down because nothing seems to go right today.",
    "The constant rain has made me feel gloomy.",
    "I am upset about the poor service at the restaurant.",
    "I didn't enjoy the movie; it was quite disappointing.",
    "My friend canceled our plans at the last minute, which hurt.",
    "I feel stressed and overwhelmed by all the work.",
    "The traffic jam made my morning extremely frustrating.",
    "I am unhappy with the results of the project.",
    "I felt let down by the cancellation of the event.",
    "I am tired of facing constant setbacks.",
    "I am depressed because I received bad news.",
    "The bitter cold made my day miserable.",
    "I feel disappointed with the lack of progress.",
    "I am upset about the argument with my colleague.",
    "I regret missing the opportunity to travel.",
    "I am frustrated by the endless delays.",
    "The noise in the city is making me anxious.",
    "I feel disheartened by the negative feedback.",
    "I am upset with the rude behavior of the staff.",
    "I feel isolated and lonely this evening.",
    "I am troubled by the constant complaints at work.",
    "I feel miserable due to the overwhelming responsibilities.",
    "I was disappointed by the lackluster performance.",
    "I feel burdened by too many problems.",
    "I am stressed about the upcoming deadlines.",
    "I feel anxious about what the future holds.",
    "I am disillusioned by the dishonesty around me.",
    "I feel hopeless after facing repeated failures.",
    "I am upset about the current state of affairs.",
    "I feel discouraged after losing the match.",
    "I am saddened by the news of the tragedy.",
    "I feel angry due to the unfair treatment I received.",
    "I am disappointed with the quality of the product.",
    "I feel frustrated because nothing is working as it should.",
    "I am dismayed by the constant negativity around me.",
    "I feel overwhelmed by all the unexpected issues.",
    "I am unhappy with how things turned out.",
    "I feel bitter about the missed opportunities.",
    "I am upset because I didn't get the promotion.",
    "I feel stressed when faced with too many choices.",
    "I am saddened by the loss of my favorite item.",
    "I feel dejected after the argument.",
    "I am worried about the uncertain future.",
    "I feel disheartened when plans fall apart.",
    "I am troubled by the lack of support.",
    "I feel miserable seeing everything go wrong.",
    "I am disappointed by the outcome of the meeting.",
    "I feel upset with the constant changes.",
    "I am frustrated with the lack of communication.",
    "I feel dejected and empty inside.",
    "I woke up feeling overwhelmed and anxious about the day ahead.",
    "The gloomy weather added to my already somber mood.",
    "I felt isolated and disconnected from those around me.",
    "A harsh remark left me feeling devalued and sad.",
    "I was disappointed by the unexpected cancellation of my plans.",
    "The constant noise in the city made me feel irritable and stressed.",
    "I felt a heavy weight on my shoulders, burdened by responsibilities.",
    "The lack of support from my colleagues left me feeling abandoned.",
    "I was let down by a friend who did not show up.",
    "I felt discouraged by the continuous setbacks in my project.",
    "The bitter cold outside made me feel miserable and lonely.",
    "I was saddened by the harsh criticism I received.",
    "The chaotic environment left me feeling drained and upset.",
    "I feel disheartened by the endless list of problems at work.",
    "I regret making a decision that led to negative outcomes.",
    "The constant conflict in my surroundings made me anxious.",
    "I felt undervalued after being overlooked for a promotion.",
    "I was dismayed by the lack of empathy from those I trusted.",
    "I felt troubled by the negative energy in the room.",
    "A series of failures left me feeling hopeless and despondent.",
    "The stressful situation made me feel trapped and helpless.",
    "I felt frustrated with the inefficiency of the service.",
    "I am disillusioned by the insincerity of people's actions.",
    "I felt the sting of rejection in a painful way.",
    "The bleak environment made it hard to see any light at the end.",
    "I felt crushed by the weight of unfulfilled expectations.",
    "The constant criticism eroded my self-confidence.",
    "I am overwhelmed by the persistent negativity in my life.",
    "I felt abandoned in a moment when I needed support.",
    "I was discouraged by the lack of progress despite my efforts.",
    "I felt bitter about being left out of important decisions.",
    "I am frustrated by the continuous delays in communication.",
    "I felt a deep sense of sorrow after the argument with a loved one.",
    "The relentless pressure made me feel defeated and exhausted.",
    "I was disheartened by the betrayal of someone I trusted.",
    "I felt ignored and unimportant in a crowded room.",
    "The uncooperative atmosphere made my day incredibly hard.",
    "I feel stressed when I face unexpected challenges.",
    "I was upset by the lack of consideration in someone's actions.",
    "I felt a deep sense of regret after a poor decision.",
    "The gloomy ambiance of the place worsened my mood.",
    "I felt despondent as nothing seemed to go my way.",
    "The ongoing argument left me emotionally drained.",
    "I was troubled by the persistent negativity that surrounded me.",
    "I felt empty after hearing disappointing news.",
    "The unfair treatment left me feeling resentful and bitter.",
    "I was disheartened by the cancellation of a long-awaited event.",
    "I felt frustrated by the constant obstacles in my path.",
    "I am deeply saddened by the recurring misfortunes in my life.",
    "I felt the sting of loss and sorrow as everything fell apart."
]


def predict(sentence):
    model.eval()
    encodings = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with th.no_grad():
        logits = model(input_ids)

    probabilities = th.softmax(logits, dim=1).squeeze()
    predicted_label = probabilities.argmax().item()

    label_mapping = {0: "Negative", 1: "Positive"}
    predicted_sentiment = label_mapping[predicted_label]

    return predicted_sentiment, probabilities

correct = 0
total = len(positive_sentences) + len(negative_sentences)

for sentence in positive_sentences:
    sentiment, _ = predict(sentence)
    if sentiment == "Positive":
        correct += 1

for sentence in negative_sentences:
    sentiment, _ = predict(sentence)
    if sentiment == "Negative":
        correct += 1

print(f"Accuracy: {correct / total * 100:.2f}%")
