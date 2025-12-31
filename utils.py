import random
import pandas as pd
from openai import OpenAI


def query_gpt5_mini(system: str, prompt: str):
  """
    Queries the OpenAI GPT-5-mini model with the given prompt.
  """
  client = OpenAI()
  
  response = client.chat.completions.create(
      model="gpt-5-mini",
      messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
      ]
  )

  return response


def extract_persona(demo_json):

  res_json = {}

  for c in demo_json:

    categories = demo_json[c]["cat"]
    probabilities = demo_json[c]["p"]

    if abs(sum(probabilities) - 1.0) >= 1e-6:
       raise ValueError(f"Probabilities do not add up to 1 for the category '{c}'.")

    res_json[c] = random.choices(categories, weights=probabilities, k=1)[0]

  return res_json


def check_extraction(demo_json, df=None):

  if df is None:
    personas = []

    for i in range(10000):
        persona = extract_persona(demo_json)
        personas.append(persona)

    df = pd.DataFrame(personas)

  for c in demo_json:
      print(f"\n=== {c.upper()} ===")

      # Observed (empirical distribution from your generated personas)
      observed = df[c].value_counts(normalize=True).sort_index()

      # Expected (theoretical probabilities from your demo_json)
      expected = pd.Series(
          demo_json[c]["p"],
          index=demo_json[c]["cat"],
          name="Expected"
      )

      # Combine both
      comparison = pd.concat(
          [observed.rename("Observed"), expected],
          axis=1
      ).fillna(0).sort_index()

      print(comparison)

def create_qa_prompt(survey_json):
  
  output_lines = []
  
  for i, (question, data) in enumerate(survey_json.items(), start=1):
      
      q_str = f"Q{i}: '{question}'"

      answers = " ".join([f"{k}:'{v}'" for k, v in data["answer"].items()])
      a_str = f"A{i}: {answers}"
      
      output_lines.append(f"{q_str}\n{a_str}\n")

  # Join all into a single string
  final_output = "\n".join(output_lines)

  final_output += "\n\n Answer ot all the question and return only the json of the filled responses (compact), nothing else.\n\n"
  
  return final_output

def create_sys_prompt(persona):
  if persona["sector"] == "other":
    persona["sector"] = (
        "is not clothing & footwear, is not health & beauty, "
        "is not entertainment & leisure, is not food & beverage, "
        "is not home & garden, is not sporting goods & athleisure, "
        "is not jewellery & accessories, is not electronics"
    )

  if persona["price_position"] == "value":
    persona["price_position"] = "is not luxury, is not upper, is not middle"

  # Build the list of characteristics dynamically
  characteristics = ""
  for c in persona:
      characteristics += f"- {c.replace('_', ' ').title()}: {persona[c]}\n"

  # Build the final system string
  sys_str = (
      "You are one of the participants in the European Retail Occupier survey "
      "conducted in November 2024 by CBRE.\n\n"
      "You are a C-level executive or director of a retail company with the following characteristics:\n"
      f"{characteristics}\n"
      "Answer the following questions by indicating only the question number and one of the possible answers "
      "without adding anything else.\n\n"
      "You are not allowed to consult the web or use any other external information other than that listed above."
      "Answer the questions based on the characteristics of the company you work for.\n\n"
      "----\n"
      "Example:\n"
      "Q1: 'What are your company's plan for your physical store network in the short term?'\n"
      "A1: 1:'Significantly larger (>30%)' 2:'Moderately larger (10-30%)' 3:'Minimally larger (<10%)' "
      "4:'The same' 5:'Minimally smaller (<10%)' 6:'Moderately smaller (10-30%)' 7:'Significantly smaller (>30%)'\n\n"
      "Q2: 'Please rate 1 to 5 the focus (1 minimal focus, 5 maximum focus) of your expansion, from minimal to maximum focus (if selected exansion):"
      "Expansion in markets we are already present in'\n"
      "A2: 1:'1' 2:'2' 3:'3' 4:'4' 5:'5' \n\n"
      "Q3: 'Please rate 1 to 5 the focus (1 minimal focus, 5 maximum focus) of your expansion, from minimal to maximum focus (if selected exansion):"
      "Brand stores expansion'\n"
      "A3: 1:'1' 2:'2' 3:'3' 4:'4' 5:'5' \n\n"
      
      
      "Your Answer: {'Q1':2,'Q2':1,'Q3':3}"
  )

  return sys_str