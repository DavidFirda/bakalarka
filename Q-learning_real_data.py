import numpy as np
import random
import matplotlib.pyplot as plt
import json
import csv
import pandas as pd

class QLearning:
    def __init__(self, categories, alpha=0.1, gamma=0.9, epsilon=0.2, epsilon_decay=0.99, q_table_file="q_table.json", log_file="question_log.csv"):
        """
        categories: Slovník kategórií
        alpha: Miera učenia (learning rate), ako rýchlo sa Q-hodnoty prispôsobujú
        gamma: Diskontný faktor (discount factor), určuje význam budúcich odmien (medzi 0 a 1)
        epsilon: Pravdepodobnosť explorácie (náhodného výberu kategórie) v epsilon-greedy stratégii
        epsilon_decay: Faktor znižovania epsilon, ktorý určuje, ako rýchlo explorácia klesá s každou iteráciou
        q_table_file: Názov súboru pre uloženie Q-tabulky
        log_file: Názov súboru, kde sa budú zaznamenávať otázky, odpovede a zmeny Q-hodnôt
        """

        self.categories = list(categories)
        self.q_table = {category: 0 for category in self.categories}
        self.correct_count = {category: 0 for category in self.categories}
        self.incorrect_count = {category: 0 for category in self.categories}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table_file = q_table_file
        self.log_file = log_file
        self.exploration_count = 0  # Počítadlo explorácie
        self.exploitation_count = 0  # Počítadlo exploitation
        self.reset_q_table()
        self.init_log_file()

    def init_log_file(self):
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID_log","Question_ID", "Category", "Answer", "Old Q-Value", "New Q-Value"])

    def save_q_table(self):
        with open(self.q_table_file, "w") as f:
            json.dump(self.q_table, f)

    def reset_q_table(self):
        self.q_table = {category: 0 for category in self.categories}
        self.save_q_table()

    def load_q_table(self):
        try:
            with open(self.q_table_file, "r") as f:
                self.q_table = json.load(f)
        except FileNotFoundError:
            self.reset_q_table()

    def update_q_value(self, question_ID, category, reward, question_order):
        old_q_value = self.q_table[category]
        max_future_q = max(self.q_table.values())
        self.q_table[category] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[category])
        new_q_value = self.q_table[category]

        answer = "Correct" if reward > 0 else "Incorrect"
        if reward > 0:
            self.correct_count[category] += 1
        else:
            self.incorrect_count[category] += 1

        self.log_interaction(question_order, question_ID, category, answer, old_q_value, new_q_value)

    def log_interaction(self, question_order, question_id, category, answer, old_q_value, new_q_value):
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([question_order, question_id, category, answer, old_q_value, new_q_value])

    def select_category(self):
        """
        Použitá stratégia je epsilon-greedy stratégia prispôsobená na zameranie sa na slabé kategórie.
        """
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.choice(self.categories)  # Exploration: skúma nové alebo náhodné kategórie bez ohľadu na ich Q-hodnoty
        else:
            self.exploitation_count += 1
            return min(self.q_table, key=self.q_table.get)  # Exploitation: Zameranie na kategóriu s najnižšou Q-hodnotou
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def get_q_table(self):
        return self.q_table
    
dataset = pd.read_csv("data/final_dataset.csv")
#print(dataset)

categories = dataset["Category"].unique().tolist() 

student = QLearning(categories, q_table_file="q_table_real.json", log_file="question_log_real.csv")

# Simulácia otázok
for iteration in range(1, 101):  # 100 iterácií
    selected_category = student.select_category()
    filtered_dataset = dataset[dataset["Category"] == selected_category]
    if filtered_dataset.empty:
        print(f"Žiadne otázky pre kategóriu: {selected_category}")
        continue
    row = filtered_dataset.sample(1).iloc[0]

    question_id = row["ID"]
    category = row["Category"]
    #subcategory = row["Subcategory"]
    correct = random.choice([True, False])
    reward = 1 if correct else -1
    student.update_q_value(question_id, category, reward, iteration)
    student.decay_epsilon()


student.save_q_table()

q_table = student.get_q_table()
categories = list(q_table.keys())
q_values = list(q_table.values())

plt.bar(categories, q_values)
plt.xlabel('Kategórie')
plt.ylabel('Q-hodnota')
plt.title('Výsledky Q-learningu podľa kategórií')
plt.xticks(rotation=90)
plt.show()

# Analýza explorácie a exploitation
total_attempts = student.exploration_count + student.exploitation_count

print(f"Celkový počet otázok: {total_attempts}")
print(f"Exploration (náhodné otázky): {student.exploration_count} ({student.exploration_count / total_attempts * 100:.2f}%)")
print(f"Exploitation (zamerané na slabé oblasti): {student.exploitation_count} ({student.exploitation_count / total_attempts * 100:.2f}%)")