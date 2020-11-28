#!/usr/bin/env python
# coding: utf-8

# In[1]:


from models import *
from utils import *
import matplotlib 
from matplotlib import pyplot as plt

# In[2]:


models_dict = {
    "Digit CNN": {"gen": get_dig_net, "aux":False, "is_digit":True, "lr" : 5e-4},
    "Simple CNN": {"gen": get_comp_net_normal, "aux": False, "is_digit":False, "lr" : 1e-3},
    "Weight sharing CNN": {"gen": get_comp_net_shared, "aux": False, "is_digit":False, "lr" : 1e-3},
    "Siamese CNN": {"gen": get_siamese_net, "aux": False, "is_digit":False, "lr" : 5e-4},
    "Auxiliary loss CNN (weight=0.5)": {"gen": get_aux_loss_cnn, "aux": True, "is_digit":False, "weight":0.5, "lr" : 5e-3},
    "Auxiliary loss CNN (weight=2)": {"gen": get_aux_loss_cnn, "aux": True, "is_digit":False, "weight":2, "lr" : 1e-3},
    "Auxiliary loss CNN (weight=10)": {"gen": get_aux_loss_cnn, "aux": True, "is_digit":False, "weight":10, "lr" : 5e-3},
    "Auxiliary loss CNN (weight=40)": {"gen": get_aux_loss_cnn, "aux": True, "is_digit":False, "weight":40, "lr" : 1e-3}
}


# In[3]:


num_rounds = 10
plt.figure()
for key in models_dict.keys():
    gen = models_dict[key]["gen"]
    aux_target = models_dict[key]["aux"]
    digit_model = models_dict[key]["is_digit"]
    aux_loss_weight = models_dict[key]["weight"] if aux_target else None
    lr = models_dict[key]["lr"]
    
    
    start = time.perf_counter()
    mean, std, mean_train, std_train, loss = evaluate_model(gen, lr = lr, is_digit_model=digit_model, use_aux_target=aux_target, rounds=num_rounds, aux_loss_weight=aux_loss_weight, iterations = 26)
    mean = 1 - mean
    mean_train = 1 - mean_train
    avg_time = (time.perf_counter() - start) / num_rounds
    plt.plot(loss, label = "{}".format(key))
    print("{} with {} rounds in {}:".format(key, num_rounds, avg_time))
    print("#parameters =", total_number_of_params(gen()))
    print("   training mean accuracy: {:.3f}".format(mean_train))
    print("   training std accuracy: {:.3f}".format(std_train))
    print("   test mean accuracy: {:.3f}".format(mean))
    print("   test stdv accuracy: {:.6f}".format(std))
    print()
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training error (%)')
plt.savefig("training.eps")

# In[ ]:




