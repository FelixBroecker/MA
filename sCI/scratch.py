def generate_combinations(lst, g):
    n = len(lst)

    # Initial combination: [0, 1, 2, ..., g-1]
    current_comb = list(range(g))
    current_combs = []
    while current_comb[0] <= n - g:
        # Create the combination based on the indices
        # combinations.append(tuple(lst[i] for i in current_comb))
        current_combs.append(current_comb)
        # Move to the next combination
        for i in range(g - 1, -1, -1):
            if current_comb[i] < n - (g - 1 - (g - 1 - i)):
                current_comb[i] += 1
                for j in range(i + 1, g):
                    current_comb[j] = current_comb[j - 1] + 1
                break
        else:
            break  # If no more valid combinations are found, stop.

    return current_combs


# Example usage
lst = [1, -1, 2, -2, 3, -3]
g = 3

result = generate_combinations(lst, g)
print(result)
