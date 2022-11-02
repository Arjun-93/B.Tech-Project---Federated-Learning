# B.Tech-Project---Federated-Learning

Link to Fed-Avg paper : https://arxiv.org/abs/1602.05629

<br>
FL Questions: 

1. Which dataset is being used? What is the total ML model accuracy that has been reported on this dataset in the centralized case?  Usually older codebases use MNIST. But nowadays, in papers, it is regular to see CIFAR 10/100 used for experiments. 
2. The details of the base ML  model being used at the clients: How was the data distributed to each of these models in the IID case ? 
3. FedAvg Behavior: What is the convergence behavior across rounds of FL in the iid case? 1 round consists of two time steps - in the first step, clients update the server and in the second step, server updates the clients. What is the batch size and number of epochs you have considered for this result?
4. How does FedAvg's accuracy behave for changes in batch size, while you keep the epochs per client constant? 
How does FedAvg's accuracy behave for constant batch size, while you change the number of epochs per client? 
5.  What is the convergence behavior of FedAvg in iid Vs non-iid case? How did you simulate the non-iid behaviour? 

