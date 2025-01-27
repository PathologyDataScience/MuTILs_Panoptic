import numpy as np
import cy_argwhere as argwhere

def test_argwhere_1d():
    # Test with a simple case
    assert argwhere.argwhere_1d([0, 1, 2, 0, 3, 0, 4]) == [1, 2, 4, 6], "Test case 1 failed."

    # Test with all zeros
    assert argwhere.argwhere_1d([0, 0, 0, 0]) == [], "Test case 2 failed."

    # Test with no zeros
    assert argwhere.argwhere_1d([1, 2, 3, 4]) == [0, 1, 2, 3], "Test case 3 failed."

    # Test with negative numbers and zeros
    assert argwhere.argwhere_1d([-1, 0, -2, 3, 0]) == [0, 2, 3], "Test case 4 failed."

    print("All tests passed!")


def test_argwhere_2d():
    assert argwhere.argwhere_2d([[0, 1, 0], [0, 0, 2], [3, 0, 0]]) == [(0, 1), (1, 2), (2, 0)], "Test case 1 failed."
    assert argwhere.argwhere_2d([[0, 0], [0, 0]]) == [], "Test case 2 failed."
    assert argwhere.argwhere_2d([[1, 2], [3, 4]]) == [(0, 0), (0, 1), (1, 0), (1, 1)], "Test case 3 failed."

    print("All tests passed!")


def test_cy_argwhere2d():
    # Define a series of test cases as boolean arrays
    test_cases = [
        np.array([[False, True, False], [True, False, True]]),
        np.array([[True, True], [True, True]]),
        np.array([[False, False], [False, False]]),
        np.random.rand(10, 10) > 0.5  # Random boolean array
    ]

    for i, test_case in enumerate(test_cases, start=1):
        # Use numpy.argwhere for the expected result
        expected = np.argwhere(test_case)
        # Use your Cython function for the actual result
        actual = argwhere.cy_argwhere2d(test_case)

        # Check if the results match, using np.array_equal to compare
        assert np.array_equal(expected, actual), f"Test case {i} failed: expected {expected}, got {actual}"

    print("All test cases passed!")

# Run the test function
if __name__ == "__main__":
    # test_argwhere_1d()
    # test_argwhere_2d()
    test_cy_argwhere2d()