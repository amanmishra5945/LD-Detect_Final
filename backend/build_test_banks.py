"""
build_test_banks.py
Generates the rich dyslexia and dysgraphia test banks for ages 5-12.
Ensures full compliance with the required item counts and age-appropriate content.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------- DYSLEXIA TEST BANK -----------------

dyslexia_bank = {
    "metadata": {
        "version": "2.0",
        "description": "Smart Learning Disability Screening Test Bank - Dyslexia (Ages 5-12)"
    },
    "letter_recognition": {
        "beginner": [
            {"id": "lr_b_1", "target": "b", "confusable_with": "d", "category": "letter_confusion", "display": "b"},
            {"id": "lr_b_2", "target": "d", "confusable_with": "b", "category": "letter_confusion", "display": "d"},
            {"id": "lr_b_3", "target": "p", "confusable_with": "q", "category": "letter_confusion", "display": "p"},
            {"id": "lr_b_4", "target": "q", "confusable_with": "p", "category": "letter_confusion", "display": "q"},
            {"id": "lr_b_5", "target": "m", "confusable_with": "n", "category": "letter_confusion", "display": "m"},
            {"id": "lr_b_6", "target": "n", "confusable_with": "m", "category": "letter_confusion", "display": "n"},
            {"id": "lr_b_7", "target": "u", "confusable_with": "v", "category": "letter_confusion", "display": "u"},
            {"id": "lr_b_8", "target": "v", "confusable_with": "u", "category": "letter_confusion", "display": "v"},
            {"id": "lr_b_9", "target": "s", "confusable_with": "z", "category": "letter_sound", "display": "s"},
            {"id": "lr_b_10", "target": "a", "confusable_with": "e", "category": "vowel", "display": "a"},
            {"id": "lr_b_11", "target": "t", "confusable_with": "l", "category": "letter", "display": "t"},
            {"id": "lr_b_12", "target": "c", "confusable_with": "k", "category": "letter", "display": "c"}
        ],
        "elementary": [
            {"id": "lr_e_1", "target": "b", "confusable_with": "d", "category": "letter_confusion", "display": "b"},
            {"id": "lr_e_2", "target": "d", "confusable_with": "b", "category": "letter_confusion", "display": "d"},
            {"id": "lr_e_3", "target": "p", "confusable_with": "q", "category": "letter_confusion", "display": "p"},
            {"id": "lr_e_4", "target": "q", "confusable_with": "p", "category": "letter_confusion", "display": "q"},
            {"id": "lr_e_5", "target": "m", "confusable_with": "n", "category": "letter_confusion", "display": "m"},
            {"id": "lr_e_6", "target": "n", "confusable_with": "m", "category": "letter_confusion", "display": "n"},
            {"id": "lr_e_7", "target": "was", "confusable_with": "saw", "category": "word_reversal", "display": "was"},
            {"id": "lr_e_8", "target": "on", "confusable_with": "no", "category": "word_reversal", "display": "on"},
            {"id": "lr_e_9", "target": "tap", "confusable_with": "pat", "category": "word_reversal", "display": "tap"},
            {"id": "lr_e_10", "target": "net", "confusable_with": "ten", "category": "word_reversal", "display": "net"}
        ],
        "intermediate": [
            {"id": "lr_i_1", "target": "from", "confusable_with": "form", "category": "transposition", "display": "from"},
            {"id": "lr_i_2", "target": "quiet", "confusable_with": "quite", "category": "transposition", "display": "quiet"},
            {"id": "lr_i_3", "target": "felt", "confusable_with": "left", "category": "transposition", "display": "felt"},
            {"id": "lr_i_4", "target": "board", "confusable_with": "broad", "category": "transposition", "display": "board"},
            {"id": "lr_i_5", "target": "dairy", "confusable_with": "diary", "category": "transposition", "display": "dairy"},
            {"id": "lr_i_6", "target": "angel", "confusable_with": "angle", "category": "transposition", "display": "angel"},
            {"id": "lr_i_7", "target": "split", "confusable_with": "spilt", "category": "transposition", "display": "split"},
            {"id": "lr_i_8", "target": "trial", "confusable_with": "trail", "category": "transposition", "display": "trial"}
        ],
        "advanced": [
            {"id": "lr_a_1", "target": "except", "confusable_with": "expect", "category": "complex_transposition", "display": "except"},
            {"id": "lr_a_2", "target": "casual", "confusable_with": "causal", "category": "complex_transposition", "display": "casual"},
            {"id": "lr_a_3", "target": "martial", "confusable_with": "marital", "category": "complex_transposition", "display": "martial"},
            {"id": "lr_a_4", "target": "desert", "confusable_with": "dessert", "category": "visual_similarity", "display": "desert"},
            {"id": "lr_a_5", "target": "adapt", "confusable_with": "adopt", "category": "vowel_shift", "display": "adapt"},
            {"id": "lr_a_6", "target": "precede", "confusable_with": "proceed", "category": "prefix_confusion", "display": "precede"},
            {"id": "lr_a_7", "target": "statute", "confusable_with": "statue", "category": "visual_similarity", "display": "statute"},
            {"id": "lr_a_8", "target": "perceive", "confusable_with": "receive", "category": "orthographic", "display": "perceive"}
        ]
    },
    "word_recognition": {
        "beginner": [
            "cat", "sun", "dog", "cup", "bed", "red", "big", "run", "hat", "pen",
            "box", "pig", "fox", "bus", "leg", "top", "van", "nut", "jam", "pin",
            "bat", "frog", "milk", "star", "nest"
        ],
        "elementary": [
            "tree", "bird", "fish", "jump", "blue", "home", "play", "fast", "wind", "cold",
            "milk", "green", "hand", "door", "boat", "rock", "desk", "duck", "sing", "walk",
            "lamp", "ring", "bear", "lion", "cake", "bell", "rain"
        ],
        "intermediate": [
            "school", "friend", "window", "garden", "yellow", "sister", "winter", "silver", "rabbit", "pocket",
            "market", "summer", "forest", "picnic", "butter", "pencil", "doctor", "ladder", "basket", "candle",
            "monkey", "kitten", "blanket", "morning", "journey", "castle", "shadow", "planet", "bridge", "guitar",
            "stream", "circle"
        ],
        "advanced": [
            "beautiful", "different", "important", "discover", "mountain", "calendar", "remember", "adventure", "together", "umbrella",
            "telescope", "creativity", "instrument", "celebrate", "knowledge", "mysterious", "experiment", "government", "curiosity", "champion",
            "neighbour", "dictionary", "whispering", "orchestra", "invention", "passenger", "landscape", "lightning", "signature", "frequency",
            "silhouette", "atmosphere"
        ]
    },
    "oral_reading": {
        "beginner": [
            {"id": "or_b_1", "text": "The little dog runs fast.", "word_count": 5},
            {"id": "or_b_2", "text": "I see a big red ball.", "word_count": 6},
            {"id": "or_b_3", "text": "The cat sits on the mat.", "word_count": 6},
            {"id": "or_b_4", "text": "A bird sings in the tree.", "word_count": 6},
            {"id": "or_b_5", "text": "We go to the park today.", "word_count": 6},
            {"id": "or_b_6", "text": "Look at the warm yellow sun.", "word_count": 6},
            {"id": "or_b_7", "text": "Sam has a cute small puppy.", "word_count": 6},
            {"id": "or_b_8", "text": "The frog jumps into the pond.", "word_count": 6},
            {"id": "or_b_9", "text": "She has ten bright shiny stars.", "word_count": 6},
            {"id": "or_b_10", "text": "My blue hat is on the bed.", "word_count": 7},
            {"id": "or_b_11", "text": "The brown duck swims in water.", "word_count": 6},
            {"id": "or_b_12", "text": "Dad bakes a sweet apple pie.", "word_count": 6}
        ],
        "elementary": [
            {"id": "or_e_1", "text": "Arun finds a tiny seed in the soft soil of the garden.", "word_count": 12},
            {"id": "or_e_2", "text": "Two brown rabbits hop quietly through the tall green grass.", "word_count": 10},
            {"id": "or_e_3", "text": "Meera reads an exciting book about brave animals at bedtime.", "word_count": 10},
            {"id": "or_e_4", "text": "The children build a tall sandcastle near the blue ocean.", "word_count": 10},
            {"id": "or_e_5", "text": "Rohan rides his shiny yellow bicycle down the quiet hill.", "word_count": 10},
            {"id": "or_e_6", "text": "A gentle breeze shakes the leaves off the cherry tree.", "word_count": 10},
            {"id": "or_e_7", "text": "The clever squirrel hides acorns under the dry brown leaves.", "word_count": 10},
            {"id": "or_e_8", "text": "Our class visited a busy bakery to watch bread bake.", "word_count": 10},
            {"id": "or_e_9", "text": "Deepak painted a picture of two playful dolphins jumping high.", "word_count": 10},
            {"id": "or_e_10", "text": "Every morning my grandfather feeds the sparrows on the porch.", "word_count": 10},
            {"id": "or_e_11", "text": "The warm soup smelled delicious on a cold rainy evening.", "word_count": 10},
            {"id": "or_e_12", "text": "Sara packed her lunchbox with sweet grapes and fresh cheese.", "word_count": 10}
        ],
        "intermediate": [
            {"id": "or_i_1", "text": "Kabir visited the regional science museum where he observed an automatic robot following a marked line.", "word_count": 16},
            {"id": "or_i_2", "text": "During their summer vacation the whole family hiked up a rocky trail to reach the sparkling mountain waterfall.", "word_count": 18},
            {"id": "or_i_3", "text": "The young astronomer pointed her telescope toward the night sky to admire the bright rings surrounding Saturn.", "word_count": 17},
            {"id": "or_i_4", "text": "After school Priya gathered several colorful autumn leaves to create an artistic collage for her biology project.", "word_count": 17},
            {"id": "or_i_5", "text": "A swift river carved a deep canyon through ancient red sandstone over thousands of quiet undisturbed years.", "word_count": 17},
            {"id": "or_i_6", "text": "The curious puppy discovered an antique copper bell buried beneath the roots of the giant willow tree.", "word_count": 17},
            {"id": "or_i_7", "text": "Our soccer team practiced diligently every afternoon until everyone mastered the difficult corner kick strategy.", "word_count": 15},
            {"id": "or_i_8", "text": "Heavy thunder echoed across the valley while bright lightning illuminated the dark clouds hovering above.", "word_count": 15},
            {"id": "or_i_9", "text": "An experienced pilot safely guided the airplane through the turbulent cloud layer toward the brightly lit runway.", "word_count": 17},
            {"id": "or_i_10", "text": "Engineers designed a modern solar panel system that efficiently powers the community library throughout the year.", "word_count": 16},
            {"id": "or_i_11", "text": "Maya carefully adjusted the strings of her acoustic violin before stepping onto the stage for the competition.", "word_count": 17},
            {"id": "or_i_12", "text": "Volunteers collected plastic containers along the sandy shoreline to protect local marine wildlife from pollution.", "word_count": 15}
        ],
        "advanced": [
            {"id": "or_a_1", "text": "Modern technological advancements have enabled researchers to study intricate deep-sea marine ecosystems that were once completely inaccessible.", "word_count": 16},
            {"id": "or_a_2", "text": "The conservationist explained that preserving biodiversity is essential for sustaining ecological stability and mitigating the consequences of climate change.", "word_count": 17},
            {"id": "or_a_3", "text": "Historic architecture reflects both aesthetic mastery and pragmatic engineering solutions devised to withstand severe earthquakes and environmental decay.", "word_count": 16},
            {"id": "or_a_4", "text": "The international committee deliberated meticulously before establishing unanimous guidelines regarding renewable energy deployment in developing regions.", "word_count": 15},
            {"id": "or_a_5", "text": "Astronomers utilized high-resolution space observatories to detect minute orbital fluctuations caused by undiscovered celestial bodies.", "word_count": 14},
            {"id": "or_a_6", "text": "A comprehensive understanding of cognitive linguistics illustrates how early phonological processing correlates directly with subsequent reading fluency.", "word_count": 16},
            {"id": "or_a_7", "text": "The ambitious student presented a sophisticated computational simulation demonstrating fluid dynamics during aircraft wing turbulence.", "word_count": 14},
            {"id": "or_a_8", "text": "Geological core samples extracted from arctic permafrost reveal unprecedented historical data concerning atmospheric gas concentrations.", "word_count": 14},
            {"id": "or_a_9", "text": "Renewable geothermal infrastructure harnesses subterranean heat reservoirs to deliver continuous clean electricity to urban communities.", "word_count": 14},
            {"id": "or_a_10", "text": "Archaeologists uncovered preserved clay tablets containing astronomical observations recorded by ancient Mesopotamian scholars thousands of years ago.", "word_count": 16},
            {"id": "or_a_11", "text": "Collaborative scientific initiatives facilitate rapid breakthroughs by integrating computational algorithms with empirical laboratory observations across continents.", "word_count": 15},
            {"id": "or_a_12", "text": "Subtle physiological adaptations allow high-altitude organisms to survive efficiently under circumstances of reduced oxygen saturation.", "word_count": 14}
        ]
    },
    "phonological_awareness": {
        "beginner": [
            {"id": "pa_b_1", "task": "Which word starts with the same sound as CAT?", "type": "first_sound", "options": ["Car", "Dog", "Sun"], "answer": 0},
            {"id": "pa_b_2", "task": "Which word rhymes with BAT?", "type": "rhyme", "options": ["Hat", "Pig", "Cup"], "answer": 0},
            {"id": "pa_b_3", "task": "Which word ends with the sound /t/?", "type": "last_sound", "options": ["Nut", "Sun", "Run"], "answer": 0},
            {"id": "pa_b_4", "task": "Which word starts with the same sound as SUN?", "type": "first_sound", "options": ["Sock", "Bed", "Fox"], "answer": 0},
            {"id": "pa_b_5", "task": "Which word rhymes with PIN?", "type": "rhyme", "options": ["Win", "Top", "Leg"], "answer": 0},
            {"id": "pa_b_6", "task": "What word do these sounds make: /d/ /o/ /g/?", "type": "blending", "options": ["Dog", "Dig", "Dot"], "answer": 0},
            {"id": "pa_b_7", "task": "Which word starts with the sound /b/?", "type": "first_sound", "options": ["Ball", "Tall", "Fall"], "answer": 0},
            {"id": "pa_b_8", "task": "Which word rhymes with RED?", "type": "rhyme", "options": ["Bed", "Hop", "Sit"], "answer": 0},
            {"id": "pa_b_9", "task": "Which word ends with the sound /g/?", "type": "last_sound", "options": ["Pig", "Pin", "Pit"], "answer": 0},
            {"id": "pa_b_10", "task": "What word do these sounds make: /s/ /u/ /n/?", "type": "blending", "options": ["Sun", "Sam", "Sin"], "answer": 0},
            {"id": "pa_b_11", "task": "Which word starts with the same sound as FROG?", "type": "first_sound", "options": ["Fish", "Duck", "Bear"], "answer": 0},
            {"id": "pa_b_12", "task": "Which word rhymes with HOP?", "type": "rhyme", "options": ["Top", "Tap", "Tip"], "answer": 0}
        ],
        "elementary": [
            {"id": "pa_e_1", "task": "Which word rhymes with TRAIN?", "type": "rhyme", "options": ["Rain", "Tree", "Track"], "answer": 0},
            {"id": "pa_e_2", "task": "What word is left if you take away /s/ from SPOT?", "type": "deletion", "options": ["Pot", "Top", "Dot"], "answer": 0},
            {"id": "pa_e_3", "task": "Which word starts with the blend /bl/?", "type": "first_sound", "options": ["Blue", "Bull", "Bell"], "answer": 0},
            {"id": "pa_e_4", "task": "Which word ends with the digraph /ch/?", "type": "last_sound", "options": ["Beach", "Beat", "Bead"], "answer": 0},
            {"id": "pa_e_5", "task": "How many syllables are in the word BUTTERFLY?", "type": "syllables", "options": ["2", "3", "4"], "answer": 1},
            {"id": "pa_e_6", "task": "Which word rhymes with LIGHT?", "type": "rhyme", "options": ["Night", "Late", "Look"], "answer": 0},
            {"id": "pa_e_7", "task": "What word is made by blending /s/ /t/ /a/ /r/?", "type": "blending", "options": ["Star", "Stir", "Scar"], "answer": 0},
            {"id": "pa_e_8", "task": "Which word starts with the blend /gr/?", "type": "first_sound", "options": ["Green", "Girl", "Game"], "answer": 0},
            {"id": "pa_e_9", "task": "What word is left if you take away /k/ from COLD?", "type": "deletion", "options": ["Old", "Hold", "Gold"], "answer": 0},
            {"id": "pa_e_10", "task": "Which word rhymes with CLOCK?", "type": "rhyme", "options": ["Rock", "Cake", "Cook"], "answer": 0},
            {"id": "pa_e_11", "task": "How many syllables are in the word ELEPHANT?", "type": "syllables", "options": ["2", "3", "4"], "answer": 1},
            {"id": "pa_e_12", "task": "Which word ends with the blend /mp/?", "type": "last_sound", "options": ["Jump", "Just", "Junk"], "answer": 0}
        ],
        "intermediate": [
            {"id": "pa_i_1", "task": "If you remove the /r/ sound from TRIP, what word remains?", "type": "deletion", "options": ["Tip", "Rip", "Tap"], "answer": 0},
            {"id": "pa_i_2", "task": "How many syllables are in the word CELEBRATION?", "type": "syllables", "options": ["3", "4", "5"], "answer": 1},
            {"id": "pa_i_3", "task": "Which word has the same vowel sound as CHAIR?", "type": "vowel_sound", "options": ["Care", "Cheer", "Car"], "answer": 0},
            {"id": "pa_i_4", "task": "What word is created by replacing /m/ in SMILE with /t/?", "type": "substitution", "options": ["Stile", "Tile", "Style"], "answer": 0},
            {"id": "pa_i_5", "task": "Which word contains three distinct syllables?", "type": "syllables", "options": ["Banana", "Apple", "Watermelon"], "answer": 0},
            {"id": "pa_i_6", "task": "Which word rhymes with BRIGHT?", "type": "rhyme", "options": ["Flight", "Bait", "Fruit"], "answer": 0},
            {"id": "pa_i_7", "task": "What word is left if you drop the /l/ from CLAP?", "type": "deletion", "options": ["Cap", "Lap", "Cup"], "answer": 0},
            {"id": "pa_i_8", "task": "Which word has the same vowel sound as BOAT?", "type": "vowel_sound", "options": ["Coat", "Boot", "Bite"], "answer": 0},
            {"id": "pa_i_9", "task": "How many speech sounds (phonemes) are in the word KNIGHT?", "type": "segmentation", "options": ["3", "4", "5"], "answer": 0},
            {"id": "pa_i_10", "task": "Change the middle vowel in BED to /a/. What word do you get?", "type": "substitution", "options": ["Bad", "Bid", "Bud"], "answer": 0},
            {"id": "pa_i_11", "task": "Which word has a silent letter?", "type": "orthography", "options": ["Wrist", "West", "Rest"], "answer": 0},
            {"id": "pa_i_12", "task": "How many syllables are in the word GEOGRAPHY?", "type": "syllables", "options": ["3", "4", "5"], "answer": 1}
        ],
        "advanced": [
            {"id": "pa_a_1", "task": "If you remove the /k/ sound from CRUMB, what word remains?", "type": "deletion", "options": ["Rum", "Rub", "Come"], "answer": 0},
            {"id": "pa_a_2", "task": "How many syllables are in the word UNFORTUNATELY?", "type": "syllables", "options": ["4", "5", "6"], "answer": 2},
            {"id": "pa_a_3", "task": "Which pair of words contains an identical root sound?", "type": "morphology", "options": ["Sign / Signal", "Site / Sight", "Son / Sun"], "answer": 0},
            {"id": "pa_a_4", "task": "What word is formed by swapping the first sounds of CAT and DOG?", "type": "spoonerism", "options": ["Dat and Cog", "Dot and Cag", "Cot and Dag"], "answer": 0},
            {"id": "pa_a_5", "task": "Which word has the primary stress on the second syllable?", "type": "stress", "options": ["Fantastic", "Camera", "Elephant"], "answer": 0},
            {"id": "pa_a_6", "task": "How many phonemes are in the word STRAIGHT?", "type": "segmentation", "options": ["5", "6", "7"], "answer": 0},
            {"id": "pa_a_7", "task": "Which word rhymes with PERSUADE?", "type": "rhyme", "options": ["Afraid", "Suede", "Provide"], "answer": 0},
            {"id": "pa_a_8", "task": "Remove the initial /s/ and final /t/ from STREET. What remains?", "type": "complex_deletion", "options": ["Tree", "Tea", "Treat"], "answer": 0},
            {"id": "pa_a_9", "task": "Which word has four distinct syllables?", "type": "syllables", "options": ["Photography", "Photographer", "Photograph"], "answer": 0},
            {"id": "pa_a_10", "task": "Identify the word that contains a diphthong sound:", "type": "phonetics", "options": ["Coin", "Can", "Cone"], "answer": 0},
            {"id": "pa_a_11", "task": "Which word does NOT rhyme with ENOUGH?", "type": "rhyme", "options": ["Through", "Tough", "Rough"], "answer": 0},
            {"id": "pa_a_12", "task": "What word is produced by reversing the sounds in /t/ /a/ /k/?", "type": "sound_reversal", "options": ["Cat", "Cut", "Cot"], "answer": 0}
        ]
    },
    "spelling": {
        "beginner": [
            {"id": "sp_b_1", "word": "cat", "hint": "A furry pet that meows"},
            {"id": "sp_b_2", "word": "sun", "hint": "It shines bright in the daytime sky"},
            {"id": "sp_b_3", "word": "dog", "hint": "A pet that wags its tail and barks"},
            {"id": "sp_b_4", "word": "red", "hint": "The color of an apple"},
            {"id": "sp_b_5", "word": "bed", "hint": "Where you sleep at night"},
            {"id": "sp_b_6", "word": "hat", "hint": "You wear it on your head"},
            {"id": "sp_b_7", "word": "pig", "hint": "A farm animal that says oink"},
            {"id": "sp_b_8", "word": "bus", "hint": "A big yellow vehicle that takes children to school"}
        ],
        "elementary": [
            {"id": "sp_e_1", "word": "tree", "hint": "It has green leaves and brown branches"},
            {"id": "sp_e_2", "word": "bird", "hint": "An animal with feathers that can fly"},
            {"id": "sp_e_3", "word": "frog", "hint": "A green creature that hops near ponds"},
            {"id": "sp_e_4", "word": "fish", "hint": "It swims in lakes and oceans"},
            {"id": "sp_e_5", "word": "hand", "hint": "It has five fingers"},
            {"id": "sp_e_6", "word": "star", "hint": "It twinkles high in the night sky"},
            {"id": "sp_e_7", "word": "milk", "hint": "A healthy white drink"},
            {"id": "sp_e_8", "word": "door", "hint": "You open it to enter a room"}
        ],
        "intermediate": [
            {"id": "sp_i_1", "word": "friend", "hint": "A pal you love to play and talk with"},
            {"id": "sp_i_2", "word": "school", "hint": "The place where you learn with teachers"},
            {"id": "sp_i_3", "word": "garden", "hint": "A yard where flowers and vegetables grow"},
            {"id": "sp_i_4", "word": "pencil", "hint": "A wooden tool used for writing and drawing"},
            {"id": "sp_i_5", "word": "window", "hint": "You look through its glass to see outside"},
            {"id": "sp_i_6", "word": "sister", "hint": "A girl who has the same parents as you"},
            {"id": "sp_i_7", "word": "yellow", "hint": "The bright cheerful color of lemons"},
            {"id": "sp_i_8", "word": "rabbit", "hint": "An animal with long ears that hops"}
        ],
        "advanced": [
            {"id": "sp_a_1", "word": "because", "hint": "A word used to explain a reason"},
            {"id": "sp_a_2", "word": "beautiful", "hint": "Pleasing to look at; gorgeous"},
            {"id": "sp_a_3", "word": "special", "hint": "Better or different from what is normal"},
            {"id": "sp_a_4", "word": "remember", "hint": "To keep something in your memory"},
            {"id": "sp_a_5", "word": "different", "hint": "Not the same as another thing"},
            {"id": "sp_a_6", "word": "together", "hint": "With each other in one group"},
            {"id": "sp_a_7", "word": "calendar", "hint": "Shows the days, weeks, and months of the year"},
            {"id": "sp_a_8", "word": "mountain", "hint": "A very tall landform rising high above the ground"}
        ]
    },
    "comprehension": {
        "beginner": [
            {
                "id": "comp_b_1",
                "passage": "Mia has a soft red ball. She rolls it across the grass to her puppy. The puppy wags its tail and runs fast.",
                "question": "What colour is Mia's ball?",
                "options": ["Red", "Blue", "Green"],
                "answer": 0
            },
            {
                "id": "comp_b_2",
                "passage": "A little yellow bird sits in the tall tree. It sings a sweet song. Ravi looks up at the branches and smiles.",
                "question": "Where is the bird sitting?",
                "options": ["In the tree", "On a car", "Inside a box"],
                "answer": 0
            },
            {
                "id": "comp_b_3",
                "passage": "The afternoon sun is warm. Ana wears a bright blue hat. She walks to the playground with her father.",
                "question": "Who goes to the playground with Ana?",
                "options": ["Her father", "Her teacher", "Her puppy"],
                "answer": 0
            },
            {
                "id": "comp_b_4",
                "passage": "Tim has a green frog. The frog likes to jump into a puddle. Tim laughs when the water splashes on his boots.",
                "question": "What animal does Tim have?",
                "options": ["A green frog", "A red bird", "A brown dog"],
                "answer": 0
            },
            {
                "id": "comp_b_5",
                "passage": "Grandma makes sweet cookies in the kitchen. She puts three cookies on a plate for Leo. Leo eats them with cold milk.",
                "question": "How many cookies are on Leo's plate?",
                "options": ["Three", "Five", "Ten"],
                "answer": 0
            }
        ],
        "elementary": [
            {
                "id": "comp_e_1",
                "passage": "Arun found a smooth brown seed in the garden soil. He planted it near the fence and watered it every morning. Within two weeks, a green sprout pushed through the earth.",
                "question": "What did Arun do every morning?",
                "options": ["He watered the seed", "He painted the fence", "He dug a new hole"],
                "answer": 0
            },
            {
                "id": "comp_e_2",
                "passage": "Meera and her brother walked to the village pond after lunch. They watched three white ducks glide smoothly across the surface. Suddenly, one duck quacked loudly and splashed their shoes.",
                "question": "How many ducks did they see at the pond?",
                "options": ["Three", "Two", "Four"],
                "answer": 0
            },
            {
                "id": "comp_e_3",
                "passage": "Rohan packed a notebook, a sharp pencil, and a fresh red apple in his backpack. He boarded the yellow bus and shared his apple with his best friend during the journey.",
                "question": "What fruit did Rohan share on the bus?",
                "options": ["An apple", "A banana", "An orange"],
                "answer": 0
            },
            {
                "id": "comp_e_4",
                "passage": "Pooja found an injured sparrow shivering on her balcony. She created a warm nest using cotton and placed small grains of rice and water nearby until the bird recovered.",
                "question": "Where did Pooja find the injured sparrow?",
                "options": ["On her balcony", "In the schoolyard", "Near the pond"],
                "answer": 0
            },
            {
                "id": "comp_e_5",
                "passage": "During arts and crafts, Dev painted a lighthouse standing on jagged rocks. He chose bright white for the tower and dark blue for the crashing sea waves.",
                "question": "What did Dev paint standing on the rocks?",
                "options": ["A lighthouse", "A windmill", "A sailboat"],
                "answer": 0
            }
        ],
        "intermediate": [
            {
                "id": "comp_i_1",
                "passage": "Kabir attended the annual inter-school science exhibition. In the robotics pavilion, an autonomous four-wheeled rover used infrared sensors to follow a curved black line taped to the floor. Kabir noted its mechanism in his journal.",
                "question": "What did the robot use to trace the curved path?",
                "options": ["Infrared sensors", "A remote control", "A tiny camera"],
                "answer": 0
            },
            {
                "id": "comp_i_2",
                "passage": "Nisha assisted her grandmother in preparing their rooftop vegetable garden. Together, they removed dry fallen leaves, enriched the soil with organic compost, and placed tomato seeds in neatly spaced rows.",
                "question": "What type of seeds did Nisha and her grandmother plant?",
                "options": ["Tomato seeds", "Sunflower seeds", "Chili seeds"],
                "answer": 0
            },
            {
                "id": "comp_i_3",
                "passage": "While organizing books in the attic on a stormy day, Dev uncovered a faded parchment map tucked inside an encyclopedia. The hand-drawn map indicated a secret trail connecting the old stone well to a historic banyan tree.",
                "question": "Where was the faded map discovered?",
                "options": ["Inside an encyclopedia in the attic", "Under a wooden desk", "Near the old well"],
                "answer": 0
            },
            {
                "id": "comp_i_4",
                "passage": "The astronomy club gathered at midnight on the open terrace. Using a motorized telescope, the teacher aligned the lens toward Saturn, revealing its brilliant ring system clearly against the pitch-black backdrop.",
                "question": "Which planet's rings did the astronomy club observe?",
                "options": ["Saturn", "Jupiter", "Mars"],
                "answer": 0
            },
            {
                "id": "comp_i_5",
                "passage": "Engineers from the municipality installed fifty solar panels on the roof of the community center. By capturing sunlight during the day, the batteries store sufficient electricity to power the facility throughout rainy evenings.",
                "question": "Where were the solar panels installed?",
                "options": ["On the community center roof", "In an open field", "Near the highway"],
                "answer": 0
            }
        ],
        "advanced": [
            {
                "id": "comp_a_1",
                "passage": "Marine biologists exploring deep oceanic trenches discovered unique ecosystems clustered around hydrothermal vents. These organisms thrive in total darkness by deriving metabolic energy from chemical synthesis rather than solar photosynthesis.",
                "question": "How do organisms near hydrothermal vents obtain their energy?",
                "options": ["Through chemical synthesis", "Through direct sunlight", "By consuming land vegetation"],
                "answer": 0
            },
            {
                "id": "comp_a_2",
                "passage": "During the archaeological survey of an ancient city, researchers unearthed an intricate subterranean aqueduct engineered from interlocking limestone blocks. The gradient maintained a steady two-degree slope to transport mountain spring water across ten kilometers without mechanical pumps.",
                "question": "What primary engineering technique allowed the aqueduct to transport water across long distances?",
                "options": ["A calculated continuous two-degree slope", "Steam-powered mechanical pumps", "Pressurized bronze pipes"],
                "answer": 0
            },
            {
                "id": "comp_a_3",
                "passage": "Contemporary cognitive research indicates that phonological processing speed in early childhood correlates significantly with reading automaticity. Targeted multisensory phonics interventions have been shown to restructure neural pathways responsible for decoding orthographic patterns.",
                "question": "According to the passage, what effect do targeted multisensory interventions have?",
                "options": ["They help restructure neural decoding pathways", "They eliminate the need for spoken language", "They replace visual letter recognition"],
                "answer": 0
            },
            {
                "id": "comp_a_4",
                "passage": "Environmental scientists monitoring arctic permafrost observed that seasonal thaw cycles have accelerated over recent decades. Deep ice core extractions enable climatologists to measure historical atmospheric gas concentrations spanning hundreds of thousands of years.",
                "question": "What information do ice core extractions provide to climatologists?",
                "options": ["Historical atmospheric gas concentrations", "Locations of subterranean rivers", "Estimates of future ocean depths"],
                "answer": 0
            },
            {
                "id": "comp_a_5",
                "passage": "The restoration committee employed non-invasive ultrasonic scanning to analyze centuries-old wooden beams supporting the cathedral dome. This diagnostic technique exposed hidden fungal degradation without compromising the architectural integrity of the landmark.",
                "question": "Why did the committee choose ultrasonic scanning for the wooden beams?",
                "options": ["It identified internal decay without damaging the structure", "It allowed workers to replace the dome immediately", "It increased the structural weight limit"],
                "answer": 0
            }
        ]
    },
    "rapid_naming": {
        "beginner": [
            {"id": "rn_b_1", "item": "A", "type": "letter"},
            {"id": "rn_b_2", "item": "O", "type": "letter"},
            {"id": "rn_b_3", "item": "S", "type": "letter"},
            {"id": "rn_b_4", "item": "Red", "type": "color"},
            {"id": "rn_b_5", "item": "Blue", "type": "color"},
            {"id": "rn_b_6", "item": "Sun", "type": "object"},
            {"id": "rn_b_7", "item": "Cat", "type": "object"},
            {"id": "rn_b_8", "item": "1", "type": "digit"}
        ],
        "elementary": [
            {"id": "rn_e_1", "item": "D", "type": "letter"},
            {"id": "rn_e_2", "item": "P", "type": "letter"},
            {"id": "rn_e_3", "item": "M", "type": "letter"},
            {"id": "rn_e_4", "item": "Green", "type": "color"},
            {"id": "rn_e_5", "item": "Yellow", "type": "color"},
            {"id": "rn_e_6", "item": "Tree", "type": "object"},
            {"id": "rn_e_7", "item": "Book", "type": "object"},
            {"id": "rn_e_8", "item": "7", "type": "digit"}
        ],
        "intermediate": [
            {"id": "rn_i_1", "item": "K", "type": "letter"},
            {"id": "rn_i_2", "item": "R", "type": "letter"},
            {"id": "rn_i_3", "item": "W", "type": "letter"},
            {"id": "rn_i_4", "item": "Orange", "type": "color"},
            {"id": "rn_i_5", "item": "Purple", "type": "color"},
            {"id": "rn_i_6", "item": "Clock", "type": "object"},
            {"id": "rn_i_7", "item": "Bridge", "type": "object"},
            {"id": "rn_i_8", "item": "9", "type": "digit"}
        ],
        "advanced": [
            {"id": "rn_a_1", "item": "Q", "type": "letter"},
            {"id": "rn_a_2", "item": "X", "type": "letter"},
            {"id": "rn_a_3", "item": "Z", "type": "letter"},
            {"id": "rn_a_4", "item": "Violet", "type": "color"},
            {"id": "rn_a_5", "item": "Silver", "type": "color"},
            {"id": "rn_a_6", "item": "Compass", "type": "object"},
            {"id": "rn_a_7", "item": "Telescope", "type": "object"},
            {"id": "rn_a_8", "item": "8", "type": "digit"}
        ]
    }
}

# ----------------- DYSGRAPHIA TEST BANK -----------------

dysgraphia_bank = {
    "metadata": {
        "version": "2.0",
        "description": "Smart Learning Disability Screening Test Bank - Dysgraphia (Ages 5-12)"
    },
    "tasks": {
        "beginner": [
            {"id": "dg_b_1", "stage": 1, "type": "letter", "target": "A", "prompt": "Write the uppercase letter A"},
            {"id": "dg_b_2", "stage": 2, "type": "letter", "target": "b", "prompt": "Write the lowercase letter b"},
            {"id": "dg_b_3", "stage": 3, "type": "simple_word", "target": "cat", "prompt": "Write the word: cat"},
            {"id": "dg_b_4", "stage": 4, "type": "simple_word", "target": "sun", "prompt": "Write the word: sun"},
            {"id": "dg_b_5", "stage": 5, "type": "phrase", "target": "red ball", "prompt": "Copy this phrase: red ball"}
        ],
        "elementary": [
            {"id": "dg_e_1", "stage": 1, "type": "letter", "target": "d", "prompt": "Write the lowercase letter d"},
            {"id": "dg_e_2", "stage": 2, "type": "letter", "target": "p", "prompt": "Write the lowercase letter p"},
            {"id": "dg_e_3", "stage": 3, "type": "word", "target": "tree", "prompt": "Write the word: tree"},
            {"id": "dg_e_4", "stage": 4, "type": "word", "target": "happy", "prompt": "Write the word: happy"},
            {"id": "dg_e_5", "stage": 5, "type": "phrase", "target": "my green garden", "prompt": "Copy this phrase: my green garden"}
        ],
        "intermediate": [
            {"id": "dg_i_1", "stage": 1, "type": "letter", "target": "q", "prompt": "Write the lowercase letter q"},
            {"id": "dg_i_2", "stage": 2, "type": "letter", "target": "m", "prompt": "Write the lowercase letter m"},
            {"id": "dg_i_3", "stage": 3, "type": "word", "target": "school", "prompt": "Write the word: school"},
            {"id": "dg_i_4", "stage": 4, "type": "word", "target": "friend", "prompt": "Write the word: friend"},
            {"id": "dg_i_5", "stage": 5, "type": "sentence", "target": "The warm sun rises early.", "prompt": "Copy this sentence: The warm sun rises early."}
        ],
        "advanced": [
            {"id": "dg_a_1", "stage": 1, "type": "letter", "target": "f", "prompt": "Write the cursive or print letter f"},
            {"id": "dg_a_2", "stage": 2, "type": "letter", "target": "k", "prompt": "Write the lowercase letter k"},
            {"id": "dg_a_3", "stage": 3, "type": "word", "target": "mountain", "prompt": "Write the word: mountain"},
            {"id": "dg_a_4", "stage": 4, "type": "word", "target": "journey", "prompt": "Write the word: journey"},
            {"id": "dg_a_5", "stage": 5, "type": "sentence", "target": "Curiosity leads to great discoveries.", "prompt": "Copy this sentence: Curiosity leads to great discoveries."}
        ]
    },
    "alternates": {
        "beginner": {
            "letters": ["A", "B", "C", "D", "b", "d", "p", "m", "t", "s"],
            "words": ["cat", "dog", "sun", "hat", "bed", "cup", "pig", "van", "box", "nut"],
            "phrases": ["red ball", "big dog", "hot sun", "my hat", "ten stars"]
        },
        "elementary": {
            "letters": ["d", "p", "q", "m", "n", "u", "v", "w", "z", "g"],
            "words": ["tree", "bird", "fish", "frog", "jump", "blue", "hand", "milk", "door", "rock"],
            "phrases": ["my green garden", "two little ducks", "a sunny morning", "playing in the park", "fresh sweet bread"]
        },
        "intermediate": {
            "letters": ["q", "m", "n", "b", "d", "h", "k", "r", "s", "t"],
            "words": ["school", "friend", "garden", "pencil", "window", "family", "yellow", "holiday", "rainbow", "guitar"],
            "sentences": [
                "The warm sun rises early.",
                "A gentle breeze moved the trees.",
                "We watched three playful dolphins.",
                "Rohan rode his new bicycle.",
                "The class visited a bakery."
            ]
        },
        "advanced": {
            "letters": ["f", "k", "z", "x", "y", "j", "g", "p", "q", "b"],
            "words": ["mountain", "journey", "beautiful", "calendar", "adventure", "together", "discovery", "signature", "creativity", "curiosity"],
            "sentences": [
                "Curiosity leads to great discoveries.",
                "Modern science explores deep marine habitats.",
                "Preserving biodiversity maintains ecological harmony.",
                "Collaborative research accelerates technological breakthroughs.",
                "Astronomers observe distant planetary formations."
            ]
        }
    }
}

# Write files
with open(DATA_DIR / "dyslexia_test_bank.json", "w", encoding="utf-8") as f:
    json.dump(dyslexia_bank, f, indent=2, ensure_ascii=False)

with open(DATA_DIR / "dysgraphia_test_bank.json", "w", encoding="utf-8") as f:
    json.dump(dysgraphia_bank, f, indent=2, ensure_ascii=False)

print("Test banks generated successfully in:", DATA_DIR)
