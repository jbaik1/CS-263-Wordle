import unittest
from wordle import Wordle

class TestWordleGame(unittest.TestCase):
    def setUp(self):
        """Initialize the game before each test."""
        self.game = Wordle()
        self.game.answer = "apple"  # Setting a known answer for consistency in tests

    def test_correct_guess(self):
        """Test that a correct guess concludes the game."""
        result = self.game.turn("apple")
        self.assertEqual(result, "GGGGG", "The guess should be entirely correct.")

    def test_wrong_guess(self):
        """Test that an incorrect guess returns the appropriate pattern."""
        result = self.game.turn("abcde")
        self.assertEqual(result, "_____", "No letters are correct or misplaced.")

    def test_partial_correct_guess(self):
        """Test guesses with partial correct letters."""
        result = self.game.turn("apric")
        self.assertEqual(result, "GG___", "First two letters are correct, others are not.")

    def test_misplaced_letters(self):
        """Test guesses with correct letters in wrong positions."""
        result = self.game.turn("leapp")
        self.assertEqual(result, "__GGG", "Last three letters are correct but misplaced.")

    def test_repeated_letters(self):
        """Test handling of repeated letters where not all occurrences are correct."""
        self.game.answer = "ballo"
        result = self.game.turn("bolls")
        self.assertEqual(result, "G_G__", "First and third letters are correct.")

    def test_game_over_after_six_guesses(self):
        """Ensure the game ends after six incorrect guesses."""
        for _ in range(6):
            self.game.turn("wrong")
        with self.assertRaises(Exception):
            self.game.turn("fails")

if __name__ == "__main__":
    unittest.main()
