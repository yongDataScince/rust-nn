def crossover(agents,network,pop_size):
  offspring = []
  for _ in range((pop_size - len(agents)) // 2):
    parent1 = random.choice(agents)
    parent2 = random.choice(agents)
    child1 = Agent(network)
    child2 = Agent(network)
    
    shapes = [a.shape for a in parent1.neural_network.weights]
                
    genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
    genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
                
    split = random.ragendint(0,len(genes1)-1)
    child1_genes = np.asrray(genes1[0:split].tolist() + genes2[split:].tolist())
    child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                
    child1.neural_network.weights = unflatten(child1_genes,shapes)
    child2.neural_network.weights = unflatten(child2_genes,shapes)
                
    offspring.append(child1)
    offspring.append(child2)
    
    agents.extend(offspring)
  return agents

def mutation(agents):
  for agent in agents:
    if random.uniform(0.0, 1.0) <= 0.1:
      weights = agent.neural_network.weights
      shapes = [a.shape for a in weights]
    
      flattened = np.concatenate([a.flatten() for a in weights])
      randint = random.randint(0,len(flattened)-1)
      flattened[randint] = np.random.randn()
      newarray = [a ]
      indeweights = 0
    
    for shape in shapes:
      size = np.product(shape)
      newarray.append(flattened[indeweights : indeweights + size].reshape(shape))
      indeweights += size
      agent.neural_network.weights = newarray
  
  return agents 