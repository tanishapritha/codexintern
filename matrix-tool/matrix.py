import numpy as np

def input_matrix(name):
    print(f"\nEnter dimensions for Matrix {name} (e.g., 2 2 for 2x2):")
    rows, cols = map(int, input().split())
    print(f"Enter values for Matrix {name} row by row, space-separated:")
    data = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != cols:
            print("Invalid number of elements. Try again.")
            return input_matrix(name)
        data.append(row)
    return np.array(data)

def print_menu():
    print("\nMatrix Operations Menu:")
    print("1. Addition (A + B)")
    print("2. Subtraction (A - B)")
    print("3. Multiplication (A x B)")
    print("4. Transpose of A and B")
    print("5. Determinant of A and B")
    print("6. Exit")

def main():
    print("Matrix Operations Tool (using NumPy)")

    A = input_matrix('A')
    B = input_matrix('B')

    while True:
        print_menu()
        choice = input("Choose an operation (1-6): ")

        if choice == '1':
            if A.shape == B.shape:
                print("\nResult of A + B:\n", A + B)
            else:
                print("Addition not possible: Matrices must be of same dimensions.")

        elif choice == '2':
            if A.shape == B.shape:
                print("\nResult of A - B:\n", A - B)
            else:
                print("Subtraction not possible: Matrices must be of same dimensions.")

        elif choice == '3':
            if A.shape[1] == B.shape[0]:
                print("\nResult of A x B:\n", np.matmul(A, B))
            else:
                print("Multiplication not possible: Columns of A must match rows of B.")

        elif choice == '4':
            print("\nTranspose of A:\n", A.T)
            print("\nTranspose of B:\n", B.T)

        elif choice == '5':
            if A.shape[0] == A.shape[1]:
                print("\nDeterminant of A:", round(np.linalg.det(A), 2))
            else:
                print("Matrix A is not square, determinant not defined.")

            if B.shape[0] == B.shape[1]:
                print("Determinant of B:", round(np.linalg.det(B), 2))
            else:
                print("Matrix B is not square, determinant not defined.")

        elif choice == '6':
            print("Exiting Matrix Tool.")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
