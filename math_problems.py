"""A small, self-contained set of grade-school math word problems.

Embedded inline (with verified integer answers) so the experiments need no
dataset download or network access. Style mirrors GSM8K; answers are exact.
"""

from __future__ import annotations

from ollama_utils import MathProblem

# Ordered hardest-first so that small --limit runs still exercise problems where
# greedy decoding fails and best-of-N / majority voting show a measurable gain.
PROBLEMS: list[MathProblem] = [
    MathProblem("James writes a 3-page letter to 2 different friends twice a week. How "
                "many pages does he write in a year?", 624),
    MathProblem("Mark has a garden with flowers. He planted plants of three colors. Ten "
                "of them are yellow, and there are 80% more of those in purple. There "
                "are only 25% as many green flowers as there are yellow and purple "
                "flowers. How many flowers does Mark have in his garden?", 35),
    MathProblem("Alexis is applying for a new job and bought a new set of business "
                "clothes. She spent $30 on a button-up shirt, $46 on suit pants, $38 on "
                "a suit coat, $11 on socks, and $18 on a belt. She also bought a pair of "
                "shoes, but lost the receipt. She had budgeted $200 and has $16 left. How "
                "much did the shoes cost?", 41),
    MathProblem("Ken created a care package to send to his brother. He placed a box on a "
                "scale, then poured in jelly beans to bring the weight to 2 pounds. Then "
                "he added brownies to triple the weight. Next, he added another 2 pounds "
                "of jelly beans. Finally, he added gummy worms to double the weight once "
                "again. What was the final weight of the box of goodies, in pounds?", 16),
    MathProblem("Tina makes $18.00 an hour. If she works more than 8 hours per shift, she "
                "is eligible for overtime, which is paid by your hourly wage + 1/2 your "
                "hourly wage. If she works 10 hours every day for 5 days, how much money "
                "does she make?", 990),
    MathProblem("Julie is reading a 120-page book. Yesterday she read 12 pages and "
                "today she read twice as many pages as yesterday. If she wants to read "
                "half of the remaining pages tomorrow, how many pages should she read?",
                42),
    MathProblem("Betty is saving for a $100 wallet. She has half the money she needs. "
                "Her parents give her $15 and her grandparents give her twice as much "
                "as her parents. How many more dollars does Betty need?", 5),
    MathProblem("Natalia sold clips to 48 friends in April, and then she sold half "
                "as many clips in May. How many clips did she sell altogether in "
                "April and May?", 72),
    MathProblem("A robe takes 2 bolts of blue fiber and half that much white fiber. "
                "How many bolts in total does it take?", 3),
    MathProblem("Weng earns $12 an hour for babysitting. Yesterday she babysat for "
                "50 minutes. How many dollars did she earn?", 10),
    MathProblem("A store sells apples for $2 each. If you buy 5 or more, you get $1 off "
                "each apple. How much do 6 apples cost?", 6),
    MathProblem("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
                "wednesday, he lost 2 more. How many golf balls did he have at the end of "
                "wednesday?", 33),
    MathProblem("A class has 24 students. One third of them got an A. Of the rest, half "
                "got a B. How many students got a B?", 8),
    MathProblem("There were nine computers in the server room. Five more computers were "
                "installed each day, from monday to thursday. How many computers are now "
                "in the server room?", 29),
    MathProblem("Olivia has $23. She bought five bagels for $3 each. How much money does "
                "she have left?", 8),
    MathProblem("Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
                "pieces do they have left in total?", 39),
    MathProblem("Shawn has five toys. For Christmas, he got two toys each from his mom "
                "and dad. How many toys does he have now?", 9),
    MathProblem("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
                "lollipops. How many lollipops did Jason give to Denny?", 8),
    MathProblem("There are 15 trees in a grove. Workers will plant trees today. After "
                "they are done, there will be 21 trees. How many trees did the workers "
                "plant today?", 6),
    MathProblem("If there are 3 cars in the parking lot and 2 more cars arrive, how many "
                "cars are in the parking lot?", 5),
]
