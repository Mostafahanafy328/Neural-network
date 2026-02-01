import matplotlib.pyplot as plt

def visual(loss_values, graph_names, title_name, Xlabel, Ylabel):
    plt.figure(figsize=(8, 5))
    #if we have many graphs
    if isinstance(loss_values[0], list):
        for values, name in zip(loss_values, graph_names):
            plt.plot(values, label=name)
    #if we have one graph
    else:
        plt.plot(loss_values, label=graph_names)

    plt.title(title_name)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
