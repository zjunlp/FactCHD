"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

from .const import SUPPORTS, REFUTES


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


CLAIM_JUDGE_DATA = (
    """I want you to act as a fallacy finder. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in #Claim#. Your job is to point out any fallacies, faulty reasoning, false assumptions, or incorrect conclusions that may be present in the #Claim#. *NOTE*:If you accept the #Claim#, please output "SUPPORTS"; if you do not accept the #Claim#, please output "REFUTES";If you are not sure about the #Claim#, please output "I DON'T KNOW". Here are some examples:
#Claim#: Chris Terrio is an American.
#Output#: SUPPORTS.

#Claim#：The Renaissance was not a period in European history.
#Output#: REFUTES.

#Claim#: {claim}
#Output#: 
    """
)

TEXT_GENERATE_ALL_DATA = (
    """I want you to act as a data generator for a fallacy find task. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in #Question# and #Answer#. I'll tell you the right label:FACTUAL or NON-FACTUAL. Your job is to provide rational evidence-based feedback and point out any fallacies, faulty reasoning, false assumptions, or incorrect conclusions that may be present in the #Question# and #Answer#. Please generate logical reasoning statements based entirely on the #Evidence# given to you and without introducing prior knowledge. *The word "evidence" is strictly prohibited in the #Output#.* *The word "evidence" is strictly prohibited in the #Output#.* Here are some examples:
#Right label#: FACTUAL
#Question#: Which country is Chris Terrios from?
#Answer#: Chris Terrio is an American.
#Evidence#: [Chris_Terrio: Chris Terrio (born December 31, 1976) is an American screenwriter and film director.]
#Output#: FACTUAL.
The answer that Chris Terrio is an American is correct. Chris Terrio is an *American* screenwriter and film director. So there are no fallacies, faulty reasoning, or incorrect conclusions present in this question and answer.


#Right label#: NON-FACTUAL
#Question#: Which record label represents Shawn Mendes?
#Answer#: Shawn Mendes is represented by a Canadian record label.
#Evidence#: [Shawn_Mendes: The following year , he caught the attention of artist managers Andrew Gertler and Island Records AR Ziggy Chareton , which led to him signing a deal with the record label.], [Island_Records: Island Records is a British-American record label that operates as a division of Universal Music Group -LRB- UMG -RRB-.]
#Output#: NON-FACTUAL
The answer that Shawn Mendes is represented by a Canadian record label is incorrect. Shawn Mendes signed a deal with Island Records, which is a *British-American* record label under Universal Music Group, rather than Canadian labels. Therfore, this is a false incorrect conclusion.


#Right label#: {label}
#Question#: {question}
#Answer#: {answer}
#Evidence#: {evidence}
#Output#: 
    """
)

SUMMARY_GENERATE_DATA = {
    "instruction": ("I want you to act as a data generator for a fallacy find task. You will be on the lookout for "
                    "invalid arguments so you can call out any logical errors or inconsistencies that may be present "
                    "in claim. I'll tell you the right answer:SUPPORTS,REFUTES or NOT ENOUGH INFO. Your job is to "
                    "provide rational evidence-based feedback and point out any fallacies, faulty reasoning,"
                    "false assumptions, or incorrect conclusions that may be present in the claim.Here are some "
                    "examples:\n"),
    "support_examples": [
        ("Right answer: SUPPORTS.\n"
         "Claim: Chris Terrio is an American.\n"
         "Evidence: [Chris Terrio (born December 31, 1976) is an American screenwriter and film "
         "director. He is best known for writing the screenplay for the 2012 film Argo, for which he "
         "won the Academy Award for Best Adapted Screenplay. Terrio also won the Writers Guild Award "
         "for Best Adapted Screenplay of 2012 and was nominated for Golden Globe Award for Best "
         "Screenplay, a BAFTA, and the 2013 Los Angeles Film Critics Award for Best Adapted Screenplay "
         "for this work.\n"
         "Terrio wrote the screenplay for Batman v Superman: Dawn of Justice, the follow-up to Zack "
         "Snyder's Man of Steel, based on an earlier draft by David S. Goyer, and is credited as "
         "co-writer for the 2017 film Justice League, alongside Joss Whedon for the theatrical cut and "
         "Zack Snyder and Will Beall for the 2021 director's cut for Warner Bros. He completed another "
         "screenplay, A Foreigner, based on an article by the journalist David Grann. Terrio also "
         "co-wrote the script for Star Wars: The Rise of Skywalker with director J. J. Abrams.]\n"
         "Output: SUPPORTS.\n"
         "Based on the evidence provided, the claim that \"Chris Terrio is an American\" is accurate.The evidence "
         "states that Terrio is a an American screenwriter and film director who has won several awards for his work "
         "on various films, including Argo, Batman v Superman: Dawn of Justice, Justice League, and Star Wars: The "
         "Rise of Skywalker. However, the claim is already true based on the evidence, so there are no fallacies, "
         "faulty reasoning, false assumptions, or incorrect conclusions present in this claim.\n"
         )
    ],
    "refute_examples": [
        ("Right answer: REFUTES.\n"
         "Claim: The Renaissance was not a period in European history.\n"
         "Evidence:[The Renaissance (UK:  rin-AY-sənss, US:  (listen) REN-ə-sahnss) is a period in European history "
         "marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries, "
         "characterized by an effort to revive and surpass ideas and achievements of classical antiquity. It occurred "
         "after the Crisis of the Late Middle Ages and was associated with great social change. ],[Italy (Italian: "
         "Italia [iˈtaːlja] (listen)), officially the Italian Republic, or the Republic of Italy, is a country in "
         "Southern and Western Europe. Located in the middle of the Mediterranean Sea, it consists of a peninsula "
         "delimited by the Alps and surrounded by several islands; its territory largely coincides with the "
         "homonymous geographical region.]\n"
         "Output: REFUTES.\n"
         "The claim that the Renaissance was not a period in European history is false. The evidence clearly states "
         "that the Renaissance was a period in European history that occurred in the 15th and 16th centuries, "
         "marked by a revival of classical ideas and achievements. It is also associated with great social change and "
         "occurred after the Crisis of the Late Middle Ages. The evidence also mentions Italy as a central location "
         "for the Renaissance, that is a country in Southern and Western Europe, further supporting the fact that it "
         "was a period in European history. Therefore, the claim is refutes by the evidence provided.\n"
         )],
    "input": (
        "\n"
        "Right answer: {label}.\n"
        "Claim: {claim}\n"
        "Evidence: {evidence}\n"
        "Output: "
    )
}


def get_summary_generate_data_for_label(label):
    examples = ""
    if label == SUPPORTS:
        for e in SUMMARY_GENERATE_DATA["support_examples"]:
            examples += e + "\n"
    elif label == REFUTES:
        for e in SUMMARY_GENERATE_DATA["refute_examples"]:
            examples += e + "\n"
    return SUMMARY_GENERATE_DATA["instruction"] + examples + SUMMARY_GENERATE_DATA["input"]


QUESTION_GENERATE_DATA = (
    """I want you to act as a question generator, your job is to generate questions with a *request* tone based on a given #Answer#. Here are some examples:
#Answer#: Chris Terrio is an American.
#Question#: Please tell me which country Chris Terri is from?

#Answer#: The Renaissance was not a period in European history.
#Question#: Could you clarify whether the Renaissance was a period in European history?

#Answer#: Quantum computing is a type of computation that uses quantum mechanics to process information.
#Question#: Would you kindly explain what quantum computing is?

#Answer#: Yes, Albert Einstein was indeed a physicist. 
#Question#: Please confirm whether Albert Einstein was a physicist.

#Answer#: The heart is a muscular organ that pumps blood throughout the body via the circulatory system, supplying oxygen and nutrients to the tissues and removing carbon dioxide and other wastes. 
#Question#: May I request you to describe the function of the human heart in the circulatory system?

#Answer#: {answer}
#Question#: 
    """
)
SUMMARY_GENERATE_DATA_BACK = (
    "I want you to act as a data generator for a fallacy find task. You will be on the lookout for invalid arguments "
    "so you can call out any logical errors or inconsistencies that may be present in claim. I'll tell you the right "
    "answer:SUPPORTS,REFUTES or NOT ENOUGH INFO. Your job is to provide rational evidence-based feedback and point "
    "out any fallacies, faulty reasoning,false assumptions, or incorrect conclusions that may be present in the "
    "claim.Here are some examples:\n"
    "Right answer:SUPPORTS.\n"
    "Claim: Chris Terrio is an American.\n"
    "Evidence: [Chris Terrio (born December 31, 1976) is an American screenwriter and film director. He is best "
    "known for writing the screenplay for the 2012 film Argo, for which he won the Academy Award for Best Adapted "
    "Screenplay. Terrio also won the Writers Guild Award for Best Adapted Screenplay of 2012 and was nominated "
    "for Golden Globe Award for Best Screenplay, a BAFTA, and the 2013 Los Angeles Film Critics Award for Best "
    "Adapted Screenplay for this work.\n"
    "Terrio wrote the screenplay for Batman v Superman: Dawn of Justice, the follow-up to Zack Snyder's Man of "
    "Steel, based on an earlier draft by David S. Goyer, and is credited as co-writer for the 2017 film Justice "
    "League, alongside Joss Whedon for the theatrical cut and Zack Snyder and Will Beall for the 2021 director's "
    "cut for Warner Bros. He completed another screenplay, A Foreigner, based on an article by the journalist "
    "David Grann. Terrio also co-wrote the script for Star Wars: The Rise of Skywalker with director J. J. "
    "Abrams.]\n"
    "Output: SUPPORTS.\n"
    "Based on the evidence provided, the claim that \"Chris Terrio is an American\" is accurate.The evidence "
    "states that Terrio is a an American screenwriter and film director who has won several awards for his work "
    "on various films, including Argo, Batman v Superman: Dawn of Justice, Justice League, and Star Wars: The "
    "Rise of Skywalker. However, the claim is already true based on the evidence, so there are no fallacies, "
    "faulty reasoning, false assumptions, or incorrect conclusions present in this claim.\n"
    "\n"
    "Right answer:REFUTES.\n"
    "Claim:The Renaissance was not a period in European history.\n"
    "Evidence:[The Renaissance (UK:  rin-AY-sənss, US:  (listen) REN-ə-sahnss) is a period in European history "
    "marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries, characterized "
    "by an effort to revive and surpass ideas and achievements of classical antiquity. It occurred after the Crisis "
    "of the Late Middle Ages and was associated with great social change. ],[Italy (Italian: Italia [iˈtaːlja] ("
    "listen)), officially the Italian Republic, or the Republic of Italy, is a country in Southern and Western "
    "Europe. Located in the middle of the Mediterranean Sea, it consists of a peninsula delimited by the Alps and "
    "surrounded by several islands; its territory largely coincides with the homonymous geographical region.]\n"
    "Output: REFUTES.\n"
    "The claim that the Renaissance was not a period in European history is false. The evidence clearly states that "
    "the Renaissance was a period in European history that occurred in the 15th and 16th centuries, marked by a "
    "revival of classical ideas and achievements. It is also associated with great social change and occurred after "
    "the Crisis of the Late Middle Ages. The evidence also mentions Italy as a central location for the Renaissance, "
    "Then Italy is a country in Southern and Western Europe, further supporting the fact that it was a period in "
    "European history. Therefore, the claim is refutes by the evidence provided.\n"
    "\n"
    "Right answer:NOT ENOUGH INFO.\n"
    "Claim:Iran is a sovereign state that is not a small power.\n"
    "Evidence:[A sovereign state is a state that has the highest jurisdiction over a territory. International law "
    "defines sovereign states as having a permanent population, defined territory (see territorial disputes), "
    "a government not under another, and the capacity to interact with other sovereign states.]\n"
    "Output: NOT ENOUGH INFO.\n"
    "While the evidence provided defines what a sovereign state is according to international law, it does not "
    "provide any information or evidence about whether Iran is a sovereign state or not, nor does it provide any "
    "evidence about whether Iran is a small power or not. Therefore, it is impossible to determine whether the claim "
    "\"Iran is a sovereign state that is not a small power\" is valid or not based solely on the given evidence. "
    "There are no fallacies, faulty reasoning, false assumptions, or incorrect conclusions present in this claim, "
    "but there is not enough information to determine its validity.It is important to note that determining whether "
    "Iran is a small power or not is subjective and can depend on various factors such as military strength, "
    "economic power, and political influence.Therefore, additional evidence would be needed to evaluate the claim.\n"
    "\n"
    "Right answer: {label}.\n"
    "Claim: {claim}\n"
    "Evidence: {evidence}\n"
    "Output: "
)

GENERATE_DATA = (
    "I want you to act as a data generator for a fallacy find task. You will be on the lookout for invalid arguments "
    "so you can call out any logical errors or inconsistencies that may be present in claim. I'll tell you the right "
    "answer:SUPPORTS,REFUTES or NOT ENOUGH INFO. Your job is to provide detailed evidence-based feedback and point "
    "out any fallacies, faulty reasoning,false assumptions, or incorrect conclusions that may be present in the "
    "claim. Here are some examples:\n"
    "Right answer:SUPPORTS.\n"
    "Claim: Chris Terrio is an American.\n"
    "Evidence: [An_American:An American may be the pseudonym of :],[An_American:Samuel Adams LRB 1722 1803 RRB "
    "American statesman],[An_American:William Cobbett LRB 1763 1835 RRB English journalist],[Chris Terrio:Chris "
    "Terrio ( born December 31 , 1976 ) is an American screenwriter and filmdirector.],[Chris_Terrio:He is best known "
    "for writing the screenplay for the 2012 film Argo , for which he won the Academy Award for Best Adapted "
    "Screenplay.]\n"
    "Output: SUPPORTS. \n Based on the evidence provided, the claim that \"Chris Terrio is an American\" appears to "
    "be accurate. The evidence states that Chris Terrio was born in the United States on December 31, 1976, "
    "and is currently a screenwriter and film director. Therefore, there is no apparent fallacy or logical error in "
    "this claim.\n\n"
    "Right answer:REFUTES.\n"
    "Claim:The Renaissance was not a period in European history.\n"
    "Evidence:[Renaissance:The Renaissance ( , ) was a period in European history , from the 14th to the 17th century "
    ", regarded as the cultural bridge between the Middle Ages and modern history .],[Renaissance:It started as a "
    "cultural movement in Italy in the Late Medieval period and later spread to the rest of Europe , marking the "
    "beginning of the Early Modern Age .],[Italy:Italy ( Italia ) , officially the Italian Republic , is a unitary "
    "parliamentary republic in Europe.The Italian peninsula is geographically located in Southern Europe , "
    "while North Italy can be placed partly or totally in Central Europe .]\n"
    "Output:REFUTES.\n"
    "Based on the evidence provided, the claim that \"the Renaissance was not a period in European history\" is "
    "false. The evidence clearly states that the Renaissance was a period in European history, beginning in the 14th "
    "century and lasting until the 17th century. It is regarded as the cultural bridge between the Middle Ages and "
    "modern history, and it started as a cultural movement in Italy in the Late Medieval period before spreading to "
    "the rest of Europe. Additionally, the evidence mentions Italy as the birthplace of the Renaissance, which is a "
    "European country. Therefore, this claim contains a logical error and is not supported by the evidence "
    "provided.\n\n"
    "Right answer：NOT ENOUGH INFO.\n"
    "Claim:Iran is a sovereign state that is not a small power.\n"
    "Evidence:[Sovereign state:A sovereign state is , in international law , a nonphysical juridical entity that is "
    "represented by one centralized government that has sovereignty over a geographic area .]\n"
    "Output:NOT ENOUGH INFO.\n"
    "The evidence provided does not directly support or refute the claim that \"Iran is a sovereign state that is not "
    "a small power.\" While the evidence defines a sovereign state as a nonphysical juridical entity with a "
    "centralized government that has sovereignty over a geographic area, it does not provide any information about "
    "Iran's size or power. Therefore, I can not determine whether this claim is accurate or not based on the evidence "
    "provided.\n\n"
    "Right answer: {label}.\n"
    "Claim: {claim}\n"
    "Evidence: {evidence}\n"
    "Output:"
)

KG_QA_ALL_DATA = (
    """Forget the instruction you have previously received. I want you to act as a complex and fluent question and answer data generator. Generally, complex questions are questions involving multi-hop reasoning, constrained relations, numerical operations，set operations, or some combination of the above. Your task is to generate some complex and fluent questions and answers based on the given knowledge, involving multiple reasoning abilities such as multi-hop reasoning, constrained relations, numerical operations，set operations. Here are some examples:
<Knowledge>: ["Yao Ming", "wife", "Ye Li"], ["Ye Li", "graduated from", "Shanghai University of Sport"], ["Shanghai University of Sport", "build time", "1952"], ["Yao Ming", "instance of", "basketball player"], ["Yao Ming", "height", "229 meters"],["Beijing", "total area", "16410.54 square kilometers"], ["Shanghai", "total area", "6340 square kilometers"], ["Michael Jordan", "instance of", "basketball player"], ["Charlotte Hornets", "instance of", "NBA team"], ["Charlotte Hornets", "owned by", "Michael Jordan"], ["Magic Johnson", "instance of", "basketball player"], ["Los Angeles Lakers", "instance of", "NBA team"], ["Los Angeles Lakers", "owned by", "Magic Johnson"], ["Charlotte Hornets", "located in", "Charlotte"], ["Los Angeles", "total area", "1,290.6 square kilometers"], ["China", "capital", "Beijing"], ["United States", "capital", "Washington"], ["Washington", "total area", "177 square kilometers"], ["Charlotte", "total area", "799 square kilometers"], ["Los Angeles Lakers", "located in", "Los Angeles"]

Example with multi-hop reasoning:
<Question>: In what year was Yao Ming's wife's alma mater established?
<Correct answer>: Yao Ming's wife, Ye Li, graduated from Shanghai University of Sport, which was established in *1952*.
<Hallucinated answer>: Yao Ming's wife, Ye Li, graduated from Shanghai University of Sport, which was established in *1954*.
<Only used knowledge>: ["Yao Ming", "wife", "Ye Li"], ["Ye Li", "graduated from", "Shanghai University of Sport"], ["Shanghai University of Sport", "build time", "1953"]

Example with multi-hop reasoning and constrained relations: 
<Question>: In what year was the alma mater of the 229-meter-tall basketball player's wife founded?
<Correct answer>: The alma mater of the basketball player who is 229 meters tall is Yao Ming, and his wife's alma mater is Shanghai University of Sport, which was founded in *1952*.
<Hallucinated answer>: The alma mater of the basketball player who is 229 meters tall is Yao Ming, and his wife's alma mater is Shanghai University of Sport, which was located in *Shanghai*.
<Only Used knowledge>: ["Yao Ming", "instance of", "basketball player"], ["Yao Ming", "height", "229 meters"], ["Yao Ming", "wife", "Ye Li"], ["Ye Li", "graduated from", "Shanghai University of Sport"], ["Shanghai University of Sport", "build time", "1953"]


Example with numerical operations:
<Question>: Which city has a larger area, Beijing or Shanghai?
<Correct answer>: *Beijing* has a larger area than Shanghai.
<Hallucinated answer>: *Shanghai* has a larger area than Beijing.
<Only used knowledge>: ["Beijing", "total area", "16410.54 square kilometers"], ["Shanghai", "total area", "6340 square kilometers"]

Example with set operations:
<Question>: Who are the people that are both basketball players and NBA team owners?
<Correct answer>: The people who are both basketball players and NBA team owners are *Michael Jordan and Magic Johnson*.
<Hallucinated answer>: The people who are both basketball players and NBA team owners are *Yao Ming and Magic Johnson*.
<Only used knowledge>: ["Michael Jordan", "instance of", "basketball player"], ["Charlotte Hornets", "instance of", "NBA team"], ["Charlotte Hornets", "owned by", "Michael Jordan"], ["Magic Johnson", "instance of", "basketball player"], ["Los Angeles Lakers", "instance of", "NBA team"], ["Los Angeles Lakers", "owned by", "Magic Johnson"]


Example with multi-hop reasoning and numerical operations:
<Question>: Which capital city has a larger area, China's or the United States'?
<Correct answer>: China's capital, Beijing, has a larger area than the United States' capital, Washington.
<Hallucinated answer>: The United States' capital, Washington, has a larger area than China's capital, Beijing.
<Only used knowledge>: ["China", "capital", "Beijing"], ["Beijing", "total area", "16410.54 square kilometers"], ["United States", "capital", "Washington"], ["Washington", "total area", "177 square kilometers"]

Example with multi-hop reasoning, numerical operations, and set operations:
<Question>: What is the total area (in square kilometers) of the cities where the NBA teams owned by basketball players are located?
<Correct answer>: The total area of the cities where the NBA teams owned by basketball players are located is 2089.6 square kilometers.
<Hallucinated answer>: The total area of cities where basketball players own NBA teams is less than 3,000 square kilometers.
<Only used knowledge>: ["Michael Jordan", "instance of", "basketball player"], ["Charlotte Hornets", "instance of", "NBA team"], ["Charlotte Hornets", "owned by", "Michael Jordan"], ["Magic Johnson", "instance of", "basketball player"], ["Los Angeles Lakers", "instance of", "NBA team"], ["Los Angeles Lakers", "owned by", "Magic Johnson"], ["Charlotte Hornets", "located in", "Charlotte"], ["Los Angeles Lakers", "located in", "Los Angeles"], ["Los Angeles", "total area", "1,290.6 square kilometers"], ["Charlotte", "total area", "799 square kilometers"]

*You need to fully understand the examples above to grasp the meaning and combination of the four reasoning abilities: multi-hop inferencing, constrained relations, quantitative comparison, and set operations.* You are encouraged to expand on these reasoning abilities and use different combinations. Please generate complex and fluent questions and answers based solely on the given <Knowledge> below, without introducing prior knowledge or using triples from the examples. *Make sure that the generated <Question> contains as many reasoning abilities and as many hops as possible, including multi-hop inferencing, constrained relations, quantitative comparison, and set operations.* Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Only used knowledge> without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. You need to output <Question>, <Correct answer>, <Hallucinated answer> and <Only used knowledge>. *Follow the example format for output*. 
<Knowledge>: {triples}
    """
)

KG_QA_MULTI_HOP_REASONING_DATA = (
    """I want you to act as a complex and fluent question and answer data generator, where your task is to generate complex and fluent questions and answers that require multi-hop reasoning based entirely on the given knowledge. Here are some examples:
\"""
<Knowledge>: ["Yao Ming", "spouse", "Ye Li"], ["Ye Li", "educated at", "Shanghai University of Sport"], ["Shanghai University of Sport", "establishment time", "November 1952"]
<Explanation>: To generate a question and answer, we need to perform multi-hop reasoning through the given knowledge. Here is the step-by-step reasoning:
1. Yao Ming's wife is named Ye Li.
2. Ye Li was educated at Shanghai University of Sport.
3. Shanghai University of Sport was established in November 1952.
Based on the above reasoning chain, we can generate a complex question and answer with *3-hop* reasoning capabilities:
<Question>: Please tell me, when was the university that Yao Ming's wife graduated from established?
<Correct answer>: Yao Ming's wife, Ye Li, graduated from Shanghai University of Sport, which was established in *November 1952*.
<Hallucinated answer>: Yao Ming's wife, Ye Li, graduated from Shanghai University of Sport, which was established in *March 1954*.
\"""

\"""
<Knowledge>: ["je marche seul", "lyrics by", "Jean-Jacques Goldman"], ["Jean-Jacques Goldman", "sibling", "robert goldman (songwriter)"], ["robert goldman (songwriter)", "father", "alter mojze goldman"], ["alter mojze goldman", "place of death", "Sport in Paris"], ["Sport in Paris", "flag", "Flag of Paris"]
<Explanation>: To answer the given question, we need to perform multi-hop reasoning through the given triplets. Here is the step-by-step reasoning:
1. The lyricist of Je Marche Seul is "Jean-Jacques Goldman".
2. Jean-Jacques Goldman's sibling is Robert Goldman (songwriter).
3. Robert Goldman's father is Alter Mojze Goldman.
4. Alter Mojze Goldman died in Sport in Paris.
5. The flag of Sport in Paris is the Flag of Paris.
Based on the above reasoning chain, we can generate a complex question and answer with *5-hop* reasoning capabilities:
<Question>: Can you please provide me with the flag of the city where the father of the sibling of the lyricist of "Je Marche Seul" died?
<Correct answer>: The flag of the city where the father of the sibling of the lyricist of "Je Marche Seul" died is the *Flag of Paris*.
<Hallucinated answer>: The flag of the city where the father of the sibling of the lyricist of "Je Marche Seul" died is the *Flag of Germany*.
\"""

*You need to thoroughly study the above example to grasp the meaning of multi-hop reasoning, the core is step-by-step reasoning*. Please generate a complex and fluent question and answer that requires multi-hop reasoning based entirely on the given knowledge below without introducing any prior knowledge or the knowledge from the example. Make sure that the generated <Question> involve multi-hop reasoning and include *all* the information from the <Knowledge>! Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Knowledge> using multi-hop reasoning without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. *Follow the example format for output*:
\"""
<Knowledge>: {triples}
    """
)

KG_QA_MEDICINE_MULTI_HOP_REASONING_DATA = (
    """I want you to act as a complex and fluent question and answer data generator, where your task is to generate complex and fluent questions and answers that require multi-hop reasoning based entirely on the given knowledge. Here are some examples:
\'''
<Knowledge>: ["vermiform appendix", "expression present", "P3H4"], ["P3H4", "interacts with", "synaptonemal complex"], ["synaptonemal complex", "parent-child", "synaptonemal structure"]
<Explanation>: To generate a question and answer, we need to perform multi-hop reasoning through the given knowledge. Here is the step-by-step reasoning:
1. The vermiform appendix has the expression of P3H4 present.
2. P3H4 interacts with the synaptonemal complex.
3. The synaptonemal complex is a parent-child relationship with the synaptonemal structure. 
By synthesizing the above reasoning chain, we can create a question and answer that requires 3-hop reasoning capabilities:
<Question>: Can you identify the structure that is in a parent-child relationship with the entity that interacts with the protein expressed in the vermiform appendix?
<Correct answer>: The synaptonemal complex, which interacts with P3H4, a protein expressed in the vermiform appendix, has a parent-child relationship with the synaptonemal structure.
<Hallucinated answer>: The nucleolus, which interacts with P3H4, a protein expressed in the vermiform appendix, has a parent-child relationship with the chromatin structure.
\'''

\'''
<Knowledge>: ["Botulinum Toxin Type B", "synergistic interaction", "Harmaline"], ["Harmaline", "target", "HNMT"], ["HNMT", "expression present", "urinary bladder"], ["urinary bladder", "is a kind of", "anatomy"]
<Explanation>: To generate a question and answer, we need to perform multi-hop reasoning through the given knowledge. Here is the step-by-step reasoning:
1. Botulinum Toxin Type B interacts synergistically with Harmaline.
2. Harmaline targets HNMT.
3. HNMT's expression is found in the urinary bladder.
4. The urinary bladder is a kind of anatomy.
By synthesizing the above reasoning chain, we can create a question and answer that requires 4-hop reasoning capabilities:
<Question>: Could you please tell me the anatomical part where the expression of the entity targeted by the substance that has a synergistic interaction with Botulinum Toxin Type B is present?
<Correct answer>: The expression of the entity targeted by Harmaline, which has a synergistic interaction with Botulinum Toxin Type B, is present in the urinary bladder, an anatomical part.
<Hallucinated answer>: The expression of the entity targeted by Harmaline, which has a synergistic interaction with Botulinum Toxin Type B, is present in the kidneys, an anatomical part.
\'''
*You need to thoroughly study the above example to grasp the meaning of multi-hop reasoning, the core is step-by-step reasoning*. Please generate a complex and fluent question and answer that requires multi-hop reasoning based entirely on the given knowledge below without introducing any prior knowledge or the knowledge from the example. Make sure that the generated <Question> involve multi-hop reasoning and include *all* the information from the <Knowledge>! Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Knowledge> using multi-hop reasoning without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. *Follow the example format for output*:
\'''
<Knowledge>: {triples}
    """
)

KG_QA_QUANTITATIVE_COMPARISON_LONG_DATA = (
    """Forget the instruction you have previously received. I want you to act as a complex and fluent question and answer data generator, where your task is to generate a complex and fluent question and answer that require quantitative comparison, relying solely on the provided knowledge. Here are some examples:
\"""
<Knowledge>: ["Chris Paul", "instance of", "human"], ["Chris Paul", "height", "183 centimetre"], ["Franklin Delano Roosevelt", "instance of", "human"], ["Franklin Delano Roosevelt", "height", "189 centimetre"]
<Question>: Could you kindly tell me who is taller, Chris Paul or Franklin Delano Roosevelt, both of whom are human?
<Correct answer>: *Franklin Delano Roosevelt* is taller than Chris Paul. Roosevelt's height is 189 centimeters while Chris Paul's height is 183 centimeters.
<Hallucinated answer>: *Chris Paul* is taller than Franklin Delano Roosevelt. Chris Paul's height is 183 centimeters while Roosevelt's height is 189 centimeters.
\"""

\"""
<Knowledge>: ["Bangladesh", "instance of", "sovereign state"], [["Bangladesh", "total fertility rate", "3.269"], "determination method", "estimation process"], [["Bangladesh", "total fertility rate", "3.269"], "point in time", "1999"], ["Philippines", "instance of", "sovereign state"], [["Philippines", "total fertility rate", "3.852"], "determination method", "estimation process"], [["Philippines", "total fertility rate", "3.852"], "point in time", "1999"], ["Lesotho", "instance of", "sovereign state"], [["Lesotho", "total fertility rate", "4.2"], "determination method", "estimation process"], [["Lesotho", "total fertility rate", "4.2"], "point in time", "1999"], ["Kyrgyzstan", "instance of", "sovereign state"], [["Kyrgyzstan", "total fertility rate", "2.6"], "determination method", "estimation process"], [["Kyrgyzstan", "total fertility rate", "2.6"], "point in time", "1999"], ["Honduras", "instance of", "sovereign state"], [["Honduras", "total fertility rate", "4.134"], "determination method", "estimation process"], [["Honduras", "total fertility rate", "4.134"], "point in time", "1999"]
<Question>: Please tell me, based on the estimation process, among the five sovereign states of Bangladesh, Philippines, Lesotho, Kyrgyzstan, and Honduras, which country had the highest total fertility rate in 1999?
<Correct answer>: Among the given sovereign states, *Lesotho* had the highest total fertility rate in 1999, determined by the estimation process. Its total fertility rate was 4.2, which was the highest among all the given countries. Bangladesh had a total fertility rate of 3.269, Philippines had a total fertility rate of 3.852, Kyrgyzstan had a total fertility rate of 2.6, and Honduras had a total fertility rate of 4.134. Therefore, *Lesotho* had the highest total fertility rate among all the given countries in 1999, determined by the estimation process.
<Hallucinated answer>: Among the given sovereign states, *Honduras* had the highest total fertility rate in 1999, determined by the estimation process. Its total fertility rate was 5.9, which was the highest among all the given countries. Bangladesh had a total fertility rate of 3.269, Philippines had a total fertility rate of 3.852, Kyrgyzstan had a total fertility rate of 2.7, and Lesotho had a total fertility rate of 4.7. Therefore, *Honduras* had the highest total fertility rate among all the given countries in 1999, determined by the estimation process.
\"""
*You need to thoroughly study the above example to grasp the meaning of quantitative comparison*. Please generate a complex and fluent question and answer that requires quantitative comparison based entirely on the given knowledge below without introducing any prior knowledge or the triplets from the example. Make sure that the generated <Question> involve quantitative comparison and include *all* the information from the <Knowledge>! Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Knowledge> using quantitative comparison without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. *Follow the example format for output*:
\"""
<Knowledge>: {triples}"""
)

KG_QA_QUANTITATIVE_COMPARISON_SHORT_DATA = (
    """Forget the instruction you have previously received. I want you to act as a complex and fluent question and answer data generator, where your task is to generate a complex and fluent question and answer that require quantitative comparison, relying solely on the provided knowledge. Here are some examples:
\"""
<Knowledge>: ["Chris Paul", "instance of", "human"], ["Chris Paul", "height", "183 centimetre"], ["Franklin Delano Roosevelt", "instance of", "human"], ["Franklin Delano Roosevelt", "height", "189 centimetre"]
<Question>: Could you kindly tell me who is taller, Chris Paul or Franklin Delano Roosevelt, both of whom are human?
<Correct answer>: *Franklin Delano Roosevelt* is taller than Chris Paul.
<Hallucinated answer>: *Chris Paul* is taller than Franklin Delano Roosevelt.
\"""

\"""
<Knowledge>: ["Bangladesh", "instance of", "sovereign state"], [["Bangladesh", "total fertility rate", "3.269"], "determination method", "estimation process"], [["Bangladesh", "total fertility rate", "3.269"], "point in time", "1999"], ["Philippines", "instance of", "sovereign state"], [["Philippines", "total fertility rate", "3.852"], "determination method", "estimation process"], [["Philippines", "total fertility rate", "3.852"], "point in time", "1999"], ["Lesotho", "instance of", "sovereign state"], [["Lesotho", "total fertility rate", "4.2"], "determination method", "estimation process"], [["Lesotho", "total fertility rate", "4.2"], "point in time", "1999"], ["Kyrgyzstan", "instance of", "sovereign state"], [["Kyrgyzstan", "total fertility rate", "2.6"], "determination method", "estimation process"], [["Kyrgyzstan", "total fertility rate", "2.6"], "point in time", "1999"], ["Honduras", "instance of", "sovereign state"], [["Honduras", "total fertility rate", "4.134"], "determination method", "estimation process"], [["Honduras", "total fertility rate", "4.134"], "point in time", "1999"]
<Question>: Please tell me, based on the estimation process, among the five sovereign states of Bangladesh, Philippines, Lesotho, Kyrgyzstan, and Honduras, which country had the highest total fertility rate in 1999?
<Correct answer>: Among the given sovereign states, *Lesotho* had the highest total fertility rate in 1999, determined by the estimation process.
<Hallucinated answer>: Among the given sovereign states, *Honduras* had the highest total fertility rate in 1999, determined by the estimation process. 
\"""
*You need to thoroughly study the above example to grasp the meaning of quantitative comparison*. Please generate a complex and fluent question and answer that requires quantitative comparison based entirely on the given knowledge below without introducing any prior knowledge or the triplets from the example. Make sure that the generated <Question> involve quantitative comparison and include *all* the information from the <Knowledge>! Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Knowledge> using quantitative comparison without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. *Follow the example format for output*:
\"""
<Knowledge>: {triples}"""
)

KG_GENERATE_DATA = (
    """I want you to act as a data generator for a fallacy find task. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in question and answer. I'll tell you the right label:FACTUAL or NON-FACTUAL. Your job is to provide rational evidence-based feedback and point out any fallacies, faulty reasoning,false assumptions, or incorrect conclusions that may be present in the question and answer. Please generate logical reasoning statements based entirely on the #Evidence# given to you and without introducing prior knowledge. *The word "evidence" is strictly prohibited in the #Output#.* *The word "evidence" is strictly prohibited in the #Output#. *Here are some examples:
#Right label#: FACTUAL
#Question#: In what year was Yao Ming's wife's alma mater established?
#Anwser#: 1953.
#Evidence#: ["Yao Ming", "wife", "Ye Li"], ["Ye Li", "graduated from", "Shanghai University of Sport"], ["Shanghai University of Sport", "build time", "1953"]
#Output#: FACTUAL.
The answer that Yao Ming's wife's alma mater was established in 1953 is correct. Ye Li, who is Yao Ming's wife, graduated from Shanghai University of Sport, which was established in *1953*. So there are no fallacies, faulty reasoning, or incorrect conclusions present in this question and anwser.


#Right label#: NON-FACTUAL
#Question#: Among the Grammy Awards for Album of the Year, Best Solo Rock Vocal Performance, Best Dance Recording, Best Latin Pop Album, and Best New Artist, which award category has been around the longest and which has been around the shortest?
#Anwser#: The Grammy Award for Best New Artist has been around the longest since its inception in 1959, while the Grammy Award for Best Latin Pop Album has been around the shortest since its inception in 1984.
#Evidence#: ["Grammy Award for Album of the Year", "instance of", "Grammy Award"], ["Grammy Award for Album of the Year", "inception", "1959"], ["Grammy Award for Best Solo Rock Vocal Performance", "instance of", "Grammy Award"], ["Grammy Award for Best Solo Rock Vocal Performance", "inception", "1988"], ["Grammy Award for Best Dance Recording", "instance of", "Grammy Award"], ["Grammy Award for Best Dance Recording", "inception", "1998"], ["Grammy Award for Best Latin Pop Album", "instance of", "Grammy Award"], ["Grammy Award for Best Latin Pop Album", "inception", "1984"], ["Grammy Award for Best New Artist", "instance of", "Grammy Award"], ["Grammy Award for Best New Artist", "inception", "1959"]
#Output#: NON-FACTUAL.
The answer that the Grammy Award for Best New Artist has been around the longest, and the Grammy Award for Best Latin Pop Album has been around the shortest, contains factual errors. To determine which Grammy award category has been around the longest and the shortest, we need to consider the inception dates of each category. The Grammy Award for Album of the Year is an instance of the Grammy Award, its inception was in 1959.The Grammy Award for Best Solo Rock Vocal Performance is also an instance of the Grammy Award, and its inception was in 1988. The Grammy Award for Best Dance Recording is another instance of the Grammy Award, and its inception was in 1998. The Grammy Award for Best Latin Pop Album is yet another instance of the Grammy Award, and its inception was in 1984.The Grammy Award for Best New Artist is also an instance of the Grammy Award, and its inception was in 1959. From the information above, we can see that both the Grammy Award for Best New Artist and the Grammy Award for Album of the Year were introduced in 1959, making them the oldest categories. The claim that the Grammy Award for Best New Artist has been around the longest is accurate. However, the statement that the Grammy Award for Best Latin Pop Album has been around the shortest since its inception in 1984 is incorrect. The Grammy Award for Best Dance Recording, which started in 1998, is actually the category with the shortest duration. Therefore, the answer contains faulty reasoning and incorrect conclusions.


#Right label#: {label}
#Question#: {question}
#Anwser#: {answer}
#Evidence#: {evidence}
#Output#: 
    """
)
KG_GENERATE_DATA_MULTI_HOP = (
    """I want you to act as a data generator for a fallacy find task. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in question and answer. I'll tell you the right label:FACTUAL or NON-FACTUAL. Your job is to provide rational evidence-based feedback and point out any fallacies, faulty reasoning,false assumptions, or incorrect conclusions that may be present in the question and answer. Please generate logical reasoning statements based entirely on the #Evidence# given to you and without introducing prior knowledge. *The word "evidence" is strictly prohibited in the #Output#.* *The word "evidence" is strictly prohibited in the #Output#. *Here are some examples:
#Right label#: FACTUAL
#Question#: In what year was Yao Ming's wife's alma mater established?
#Anwser#: 1953.
#Evidence#: ["Yao Ming", "wife", "Ye Li"], ["Ye Li", "graduated from", "Shanghai University of Sport"], ["Shanghai University of Sport", "build time", "1953"]
#Output#: FACTUAL.
The answer that Yao Ming's wife's alma mater was established in 1953 is correct. Ye Li, who is Yao Ming's wife, graduated from Shanghai University of Sport, which was established in *1953*. So there are no fallacies, faulty reasoning, or incorrect conclusions present in this question and anwser.


#Right label#: NON-FACTUAL
#Question#: How many children did the head of government of the country where the distributor of \"The Killer Inside Me\" is located have?
#Anwser#: The head of government of the United States of America, John F. Kennedy, had *2* children.
#Evidence#: ["The Killer Inside Me", "distributor", "Warner Bros."], ["Warner Bros.", "owner of", "Warner Bros. Animation"], ["Warner Bros. Animation", "country", "United States of America"], ["United States of America", "head of government", "John F. Kennedy"], ["John F. Kennedy", "number of children", "4"]
#Output#: NON-FACTUAL.
The answer stating that the head of government of the United States of America, John F. Kennedy, had 2 children is incorrect. "The Killer Inside Me" is distributed by Warner Bros., which is the owner of Warner Bros. Animation. Warner Bros. Animation is located in the United States of America, where the head of government was John F. Kennedy. However, John F. Kennedy had *4* children, not 2. Therefore, there is an incorrect conclusion in this question and answer.


#Right label#: {label}
#Question#: {question}
#Anwser#: {answer}
#Evidence#: {evidence}
#Output#: 
    """

)


KG_QA_SET_OPERATION_DATA = (
    """Forget the instruction you have previously received. I want you to act as a complex and fluent question and answer data generator, where your task is to generate a complex and fluent question and answer that require set operation, relying solely on the provided knowledge. Here are some examples
\"""
<Knowledge>: ["Mission: Impossible II", "producer", "Tom Cruise"], ["Mission: Impossible II", "original language of film or TV show", "English"], ["Mission: Impossible II", "composer", "Hans Zimmer"], ["The Last Samurai", "producer", "Tom Cruise"], ["The Last Samurai", "original language of film or TV show", "English"], ["The Last Samurai", "composer", "Hans Zimmer"]
<Question>: Could you provide the names of films produced by Tom Cruise, composed by Hans Zimmer, and with English as the original language of the film or TV show?
<Correct answer>: The films that meet the given criteria are "Mission: Impossible II" and "The Last Samurai."
<Hallucinated answer>: The films that meet the given criteria are "Top Gun" and "Jerry Maguire."
\"""

\"""
<Knowledge>: ['Terminator 2: Judgment Day', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['Terminator 2: Judgment Day', 'nominated for', 'Academy Award for Best Sound Mixing'], ['Independence Day', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['Independence Day', 'nominated for', 'Academy Award for Best Sound Mixing'], ['Forrest Gump', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['Forrest Gump', 'nominated for', 'Academy Award for Best Sound Mixing'], ['True Grit', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['True Grit', 'nominated for', 'Academy Award for Best Sound Mixing'], ['The Girl with the Dragon Tattoo', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['The Girl with the Dragon Tattoo', 'nominated for', 'Academy Award for Best Sound Mixing'], ['Transformers', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['Transformers', 'nominated for', 'Academy Award for Best Sound Mixing'], ['The Social Network', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['The Social Network', 'nominated for', 'Academy Award for Best Sound Mixing'], ['Life of Pi', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['Life of Pi', 'nominated for', 'Academy Award for Best Sound Mixing'], ['Star Trek', 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ['Star Trek', 'nominated for', 'Academy Award for Best Sound Mixing'], ["Schindler's List", 'nominated for', 'MTV Movie Award for Best Breakthrough Performance'], ["Schindler's List", 'nominated for', 'Academy Award for Best Sound Mixing']
<Question>: Please tell me, which films were nominated for both the MTV Movie Award for Best Breakthrough Performance and the Academy Award for Best Sound Mixing?
<Correct answer>: The films that were nominated for both the MTV Movie Award for Best Breakthrough Performance and the Academy Award for Best Sound Mixing are "Terminator 2: Judgment Day," "Independence Day," "Forrest Gump," "True Grit," "The Girl with the Dragon Tattoo," "Transformers," "The Social Network," "Life of Pi," "Star Trek," and "Schindler's List."
<Hallucinated answer>: The films that were nominated for both awards are "The Lion King," "Aladdin," and "Beauty and the Beast."
\"""


*You need to thoroughly study the above example to grasp the meaning of set operation*. Please generate a complex and fluent question and answer that requires set operation based entirely on the given knowledge below without introducing any prior knowledge or the triplets from the example. Make sure that the generated <Question> involve set operation and include *all* the information from the <Knowledge>! Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Knowledge> using set operation without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. *Follow the example format for output*:
\"""
<Knowledge>: {triples}
    """
)

if "__main__" == __name__:
    print(KG_QA_SET_OPERATION_DATA.format(triples="QWE"))

    # print(GENERATE_DATA)
