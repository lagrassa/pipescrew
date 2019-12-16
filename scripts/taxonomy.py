import pygraphviz as pgv
print("blue = question")
print("black = answer")
G = pgv.AGraph(directed = True)
G.add_node("Does a solution exist?", color='blue')
G.add_node("Human intervention", color='black')
G.add_edge("Does a solution exist?", "Human intervention", color="red", label='n')


G.add_node("Can the planner find a solution?", color = 'blue')
G.add_edge("Does a solution exist?", "Can the planner find a solution?", color="green", label='y')



G.add_node("Is finding the solution fast?", color = 'blue')
G.add_edge("Can the planner find a solution?", "Is finding the solution fast?", color="green", label='n')
 
#Reliability
G.add_node("Is the solution reliable?", color="blue")
G.add_node("Use and follow the planner", color="black")
G.add_edge("Is the solution reliable?", "Use and follow the planner", color="green", label="y")
G.add_edge("Is finding the solution fast?", "Is the solution reliable?", color = 'green', label='y')


#Dealing with time
G.add_node("Is motion stereotypical?", color= 'blue')
G.add_node("How stereotypical?", color= 'blue')
G.add_node("Can the types of trajectories be segmented?", color= 'blue')

G.add_edge("Is finding the solution fast?", "Is motion stereotypical?", color = 'red', label='n')
G.add_edge("Is motion stereotypical?", "How stereotypical?", color='green', label='y')
G.add_edge("How stereotypical?", "Fit DMP to planner output", color="green", label='very')
G.add_edge("How stereotypical?", "Reinforcement learning trained using prior trajectories", color="yellow", label='somewhat')
G.add_edge("How stereotypical?", "Can the types of trajectories be segmented?", color="red", label="not")
G.add_edge("Can the types of trajectories be segmented?", "Reinforcement learning using large model-free policy", color="red", label='n')
G.add_edge("Can the types of trajectories be segmented?", "Segment then", color="green", label='y')
G.add_edge("Segment then", "Fit DMP to planner output", color="green", label="y")


G.add_node("Reinforcement learning trained using prior trajectories", color = "black")
G.add_edge("Can the planner find a solution?", "Reinforcement learning trained using prior trajectories", color="yellow", label="sometimes")
G.add_edge("Is motion stereotypical?", "Reinforcement learning trained using prior trajectories", color="green", label='y')

#Going back to reliability
G.add_edge("Is the solution reliable?", "Is motion stereotypical?", color="red", label="n")


G.layout(prog='dot')
G.write("taxonomy.dot")
G.draw("taxonomy.png")


