
"""# Server Side Training"""

import time
import copy
from copy import deepcopy
import matplotlib.pyplot as plt

def ss_train_r(model, rounds, plt_title="Loss Curve"):

  
  # global model weights
  global_weights = model.state_dict()

  # training loss
  train_loss = []

  # measure time
  start = time.time()


  for current_round in range(1, rounds+1):

    w, local_loss = [], []

    # model1 = LogisticRegression(n_features)
    weights1, loss1 = clients_training(X_train_1, y_train_1, model)

    # model2 = LogisticRegression(n_features)
    weights2, loss2 = clients_training(X_train_2, y_train_2, model)

    w.append(copy.deepcopy(weights1))
    w.append(copy.deepcopy(weights2))
    # print(loss1)
    # print(loss2)
    local_loss.append(copy.deepcopy(loss1))
    local_loss.append(copy.deepcopy(loss2))

    # updating the global weights
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
      for i in range(1, len(w)):
        weights_avg[k] += w[i][k]

      weights_avg[k] = torch.div(weights_avg[k], len(w))
    global_weights = weights_avg

    # move the updated weights to our model state dict
    model.load_state_dict(global_weights)

    # loss
    loss_avg = sum(local_loss) / len(local_loss)
    print('Round: {}... \tAverage Loss: {}'.format(current_round, round(loss_avg, 3)))
    train_loss.append(loss_avg)

  plt.plot(train_loss)
  end = time.time()
  print("Training Done!")
  print("Total time taken to Train: {}".format(end-start))

  return model

model = LogisticRegression(n_features)

if torch.cuda.is_available():
  model.cuda()

server_training = ss_train_r(model, 100,"Loss Curve")