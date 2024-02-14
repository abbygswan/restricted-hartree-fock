# restricted-hartree-fock

This is a code that finds the ground state energy for molecules. The molecule is specified by the geometry.dat file; the one uploaded is for H2O. It makes guesses based on a density matrix, and makes a new density matrix based on that guess. The code iterates over each new density matrix until the ground state energy converges to a value that is less than 10^{-8} from the last guess. 
