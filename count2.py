# %%
import os
z=0
for dir in os.listdir(os.getcwd() + "/data"):
    print(dir)
    for objects in os.listdir(os.getcwd() + "/data/"+ dir):
            print(str(objects) + ":" , len(os.listdir(os.getcwd() + "/data/" + dir + "/" + objects)) ) 
            z+=len(os.listdir(os.getcwd() + "/data/" + dir + "/" + objects))
print(z)

# %%
