# writing yini

import pickle
list = 'ProjectName: CosmoL75n64'
f = open('L75n64_JW.yini', 'wb')

pickle.dump(list, f)
f.close()

####### reading
f = open("L75n64_JW.yini", "rb")
try:
   while True:
      s = pickle.load(f)
      print(s)
except EOFError:
    f.close()
