from reccomendation_engine.convertor import num_users, num_items, total_nodes, interactions
import random

def generate_negative_samples():
    triplets = []
    
    for i in range(num_users):
        userId = i
        
        interacted_items = []
        non_interacted_items = []
        for user, item in interactions:
            if(user == userId):
                interacted_items.append(item)

        for item_no in range(num_items):
            if item_no not in interacted_items:
                non_interacted_items.append(item_no)

        for item in interacted_items:
            triplets.append((userId, item, random.choice(non_interacted_items)))


    return triplets


# triplets = generate_negative_samples()

# print(triplets)
            
        