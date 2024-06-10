def calculate_hypervolume_cost(hypervolume, cost):

    result = hypervolume * (7 - cost / 2625000)
    return result

if __name__ == "__main__":
    
    hypervolume = float(input("Enter the Hypervolume value: "))
    cost = float(input("Enter the Cost value: "))
    output = calculate_hypervolume_cost(hypervolume, cost)
    print(f"The calculated output is: {output}")
