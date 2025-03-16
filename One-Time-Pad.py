# The goal of the small script is to implement a one-time pad encryption algorithm and to calculate the entropies of guessing a message with a uniform distribution and a known frequency per letter of the english alphabet.
# The script generates a random key of length n, encrypts a message with the key, and decrypts the encrypted message with the key.
# The script also calculates the entropy of guessing one character of the key and a 5-letter word with a uniform distribution and a known frequency per letter of the english alphabet.
# The script is based on the following resources:
# https://en.wikipedia.org/wiki/One-time_pad
# https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html (The frequency of each letter of the english alphabet)
# https://www.geeksforgeeks.org/one-time-pad-encryption-algorithm/

import scipy
import numpy as np
import random
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors 
from matplotlib import style
style.use('seaborn-v0_8-deep')

PRINT_DEBUG = False
PRINT_INFO = True
SHOW_PLOTS = True

# Message: first 2000 letters of Shannon's "A Mathematical Theory of Communication" (1948)
plainTextMessage = " The problems of cryptography and secrecy systems furnish an interesting application of communication theory.' In this paper a theory of " \
            "secrecy systems is developed. The approach is on a theoretical level and is" \
            "intended to complement the treatment found in standard works on cryptography." \
            "There, a detailed study is made of the many standard types ofcodes and ciphers, " \
            "and of the ways of breaking them. We will be more con￾cerned with the general mathematical structure and properties of secrecy " \
            "systems. The treatment is limited in certain ways. First, there are three general" \
            "types of secrecy system: (1) concealment systems, including such methods" \
            "as invisible ink, concealing a message in an innocent text, or in a fake cover￾ing cryptogram, or other methods in which the existence of the message is" \
            "concealed from the enemy; (2) privacy systems, for example speech inver￾sion, in which special equipment is required to recover the message; (3)" \
            "'true' secrecy systems where the meaning of the message is concealed by" \
            "cipher, code, etc., although its existence is not hidden, and the enemy is" \
            "assumed to have any special equipment necessary to intercept and record" \
            "the transmitted signal. We consider only the third type-concealment" \
            "systems are primarily a psychological problem, and privacy systems a technological one." \
            "Secondly, the treatment is limited to the case of discrete information," \
            "where the message to be enciphered consists of a sequence of discrete symbols, each chosen from a finite set. These symbols may be letters in a lan￾guage, words of a language, amplitude levels of a 'quantized' speech or video" \
            "signal, etc., but the main emphasis and thinking has been concerned with the case of letters." \
            "The paper is divided into three parts. The main results will now be briefly" \
            "summarized. The first part deals with the basic mathematical structure of secrecy systems" \
            "As in communication theory a language is considered to" \
            "be represented by a stochastic process which produces a discrete sequence of" \
            "symbols in accordance with some system of probabilities. Associated with a" \
            "language there is a certain parameter D which we call the redundancy of" \
            "the language. D measures, in a sense, how much a text in the language can" \
            "be reduced in length without losing any information. As a simple example," \
            "since u always follows q in English words, the u may be omitted without loss." \
            "Considerable reductions are possible in English due to the statistical struc￾ture of the language, the high frequencies of certain letters or words, etc." \
            "Redundancy is of central importance in the study of secrecy systems. j"

# Extend the plainTextMessage to x times, by repeating the message
plainTextMessage = plainTextMessage * 1

# The frequency of each letter of the english alphabet in percentage
letter_freq_ENG = { 'E': 11.1606, 'A': 8.4966, 'R': 7.5809, 'I': 7.5448, 'O': 7.1635, 'T': 6.9509, 'N': 6.6544, 
                    'S': 5.7351, 'L': 5.4893, 'C': 4.5388, 'U': 3.6308, 'D': 3.3844, 'P': 3.1671, 'M': 3.0129, 
                    'H': 3.0034, 'G': 2.4705, 'B': 2.0720, 'F': 1.8121, 'Y': 1.7779, 'W': 1.2899, 'K': 1.1016, 
                    'V': 1.0074, 'X': 0.2902, 'Z': 0.2722, 'J': 0.1965, 'Q': 0.1962 }

def entropy_calculations(msgLen):
    entropy_one_letter()
    entropy_5_letter_word()
    # Calculate the entropy of guessing a msg
    entropy_of_plain_text(msgLen)
    
# Calculate the entropy of guessing one character of the key
def entropy_one_letter():
    # Calculate the probability of guessing one character of the key with a uniform distribution
    uniform_probability = 1/26
    if PRINT_INFO: print("The probability of guessing one character of the key with a uniform distribution is: ", uniform_probability)

    # Calculate the entropy of a uniform distribution
    uniform_entropy = -np.sum((uniform_probability)*np.log2(uniform_probability)*26)
    if PRINT_INFO: print("The entropy of guessing one character of the key with a uniform distribution is: ", uniform_entropy)

    # Calculate the entropy of a known frequency per letter of the english alphabet
    # The frequency of each letter of the english alphabet
    frequency = list(letter_freq_ENG.values())
    frequency = [i/100 for i in frequency]
    entropy = -np.sum(frequency*np.log2(frequency))
    if PRINT_INFO: print("The entropy of guessing one character of the key with a known frequency per letter of the english alphabet is: ", entropy)
    if PRINT_INFO: print("The probability of guessing one character of the key with a known frequency per letter of the english alphabet is: ", 2**(-entropy))
    
# Calculate the entropy of guessing a 5-letter word
def entropy_5_letter_word():
    # Calculate the probability of guessing a 5-letter word with a uniform distribution
    uniform_probability = 1/26
    five_letter_word = uniform_probability**5
    if PRINT_INFO: print("The probability of guessing a 5-letter word with a uniform distribution is: ", five_letter_word)

    # Calculate the entropy of a uniform distribution
    uniform_entropy = -np.sum((uniform_probability)*np.log2(uniform_probability)*26)*5
    if PRINT_INFO: print("The entropy of guessing a 5-letter word with a uniform distribution is: ", uniform_entropy)

    # Calculate the probability of guessing a 5-letter word with a known frequency per letter of the english alphabet
    frequency = list(letter_freq_ENG.values())
    frequency = [i/100 for i in frequency]
    if PRINT_INFO: print("The probability of guessing a 5-letter word with a known frequency per letter of the english alphabet is: ", np.prod(frequency)**5)

    # Calculate the entropy of a known frequency per letter of the english alphabet
    # The frequency of each letter of the english alphabet
    entropy = -np.sum(frequency*np.log2(frequency))*5
    if PRINT_INFO: print("The entropy of guessing a 5-letter word with a known frequency per letter of the english alphabet is: ", entropy)

def entropy_of_plain_text(msgLen):
    # Calculate the probability of guessing a 2000-letter text with a uniform distribution
    uniform_probability = 1/26
    uniform_text_gues_prob = uniform_probability**msgLen
    if PRINT_INFO: print("The probability of guessing a " + str(msgLen) + "-letter text with a uniform distribution is: ", uniform_text_gues_prob)

    # Calculate the entropy of a uniform distribution
    uniform_entropy = -np.sum((uniform_probability)*np.log2(uniform_probability)*26)*msgLen
    if PRINT_INFO: print("The entropy of guessing a " + str(msgLen) + "-letter text with a uniform distribution is: ", uniform_entropy)

    # Calculate the probability of guessing a msgLen-letter text with a known frequency per letter of the english alphabet
    frequency = list(letter_freq_ENG.values())
    frequency = [i/100 for i in frequency]

    # Calculate the entropy of a known frequency per letter of the english alphabet
    # The frequency of each letter of the english alphabet
    entropy = -np.sum(frequency*np.log2(frequency))*msgLen
    if PRINT_INFO: print("The entropy of guessing a " + str(msgLen) + "-letter text with a known frequency per letter of the english alphabet is: ", entropy)
    if PRINT_INFO: print("The probability of guessing a " + str(msgLen) + "-letter text with a known frequency per letter of the english alphabet is: ", 2**(-entropy))

# One-Time Pad Encryption Algorithm
def OneTimePadEncryptionAlgorithm():
    # Process a message of 2000 letters
    message = process_message(plainTextMessage)
    msgLen = len(message)
    if PRINT_DEBUG: print("The processed message is: ", message)
    
    # Generate a random key of length msgLen
    key = generate_key(msgLen)
    if PRINT_DEBUG: print("The generated key is: ", key)

    # Encrypt a message with the key
    cypher = encrypt(message, key)
    if PRINT_DEBUG: print("The encrypted message is: ", cypher)

    # Decrypt the encrypted message with the key
    decrypted_message = decrypt(cypher, key)
    if PRINT_DEBUG: print("The decrypted message is: ", decrypted_message)

    # Check if the decrypted message is equal to the original message
    if message == "".join(decrypted_message):
        print("The decrypted message is equal to the original message.")
    else:
        print("The decrypted message is not equal to the original message.")

    return message, cypher, key

# Process the message by removing the spaces, symbols and numbers and converting the message to uppercase, then count the length
def process_message(message):
    # Remove the spaces
    message = message.replace(" ", "")
    
    # Remove the symbols and numbers
    message = ''.join(e for e in message if e.isalpha())
    
    # Convert the message to uppercase
    message = message.upper()
    
    # Count the characters
    if PRINT_DEBUG: print("The length of the message is: ", len(message))
    return message

# Calculate the probability distribution of a message
def probability_distribution(message):
    frequency = {}
    for i in message:
        if i in frequency:
            frequency[i] += 1
        else:
            frequency[i] = 1

    for i in frequency:
        frequency[i] = frequency[i] / len(message)

    # Print the frequency of each letter of the message
    if PRINT_DEBUG: print("The frequency of each letter of the message is: ", frequency)
    return frequency

# Generate a random key of length n
def generate_key(n):
    key = []
    for i in range(n):
        key.append(chr(random.randint(65, 90)))
    return key

# Encrypt a message with a key
def encrypt(message, key):
    encrypted_message = []
    for i in range(len(message)):
        encrypted_message.append(chr((ord(message[i]) + ord(key[i])) % 26 + 65))

    return encrypted_message

# Decrypt a message with a key
def decrypt(encrypted_message, key):
    decrypted_message = []
    for i in range(len(encrypted_message)):
        decrypted_message.append(chr((ord(encrypted_message[i]) - ord(key[i])) % 26 + 65))

    return decrypted_message

def calculate_joint_independent_pmf(pdf_msg, pdf_cypher):
    pdf_cypher_arr = np.array(list(pdf_cypher.values()))[:,np.newaxis]
    pdf_arr = np.array(list(pdf_msg.values()))[:,np.newaxis]
    if PRINT_DEBUG: print("Shape of pdf_cypher: ", pdf_cypher_arr.shape)
    if PRINT_DEBUG: print("Shape of pdf_msg: ", pdf_arr.shape)

    # Calculate the joint pdf of the cypher and the prior distribution of the message
    pdf_joint_indep_arr = pdf_cypher_arr @ pdf_arr.T
    if PRINT_DEBUG: print("Shape of pdf_joint: ", pdf_joint_indep_arr.shape)
    if PRINT_DEBUG: print("rank of pdf_joint: ", np.linalg.matrix_rank(pdf_joint_indep_arr))

    # Normalize to get joint pmf
    joint_pmf = pdf_joint_indep_arr / np.sum(pdf_joint_indep_arr)
    if PRINT_DEBUG: print("Sum of joint pmf: ", np.sum(joint_pmf))
    if PRINT_DEBUG: print("rank of joint_pmf: ", np.linalg.matrix_rank(joint_pmf))
    
    # sort the dictionary by the alphabet
    joint_pmf = dict(zip(list(pdf_msg.keys()), joint_pmf))
    return joint_pmf

def calculate_joint_dependent_pmf(message, cypher, keys):
    
    # convert message string to array and cypher character array to number array from 0 to 25
    message_arr = np.array([ord(c) - ord('A') for c in message])
    cypher_arr = np.array([ord(c) - ord('A') for c in cypher])
    if PRINT_DEBUG: print("Shape of message_arr: ", message_arr.shape)
    if PRINT_DEBUG: print("Shape of cypher_arr: ", cypher_arr.shape)

    # Compute joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(message_arr.squeeze(), cypher_arr.squeeze(), bins=26, range=[[0, 25], [0, 25]])
    if PRINT_DEBUG: print("Shape of hist_2d: ", hist_2d.shape)
    if PRINT_DEBUG: print("Sum of hist_2d: ", np.sum(hist_2d))
    if PRINT_DEBUG: print("Rank of hist_2d: ", np.linalg.matrix_rank(hist_2d))
    if PRINT_DEBUG: print("hist_2d: ", hist_2d)

    # Normalize to get joint probability P(C, M)
    joint_pmf = hist_2d / np.sum(hist_2d)
    joint_pmf = joint_pmf.T

    # sort the dictionary by the alphabet
    joint_pmf = dict(zip(list(keys), joint_pmf))

    return joint_pmf

def calculate_eigenvalues(joint_pmf):
    # extract the eigenvalues of the joint pmf
    joint_pmf = np.array(list(joint_pmf.values()))
    eigenvalues = np.linalg.eigvals(joint_pmf)
    eigenvalues = np.abs(eigenvalues)    # only real and positive part of the eigenvalues
    eigenvalues = np.sort(eigenvalues)[::-1]
    if PRINT_DEBUG: print("Eigenvalues of the joint pmf: ", eigenvalues)
    
    # print the spectral gap, condition number, dominance ratio and effective rank of the joint pmf
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    condition_number = eigenvalues[0] / eigenvalues[-1]
    dominance_ratio = eigenvalues[0] / np.sum(eigenvalues)
    effective_rank = eigenvalues[0] / (np.sum(eigenvalues)-eigenvalues[0])
    if PRINT_INFO: print("Eigenvalue stats for msgLen")
    if PRINT_INFO: print("Spectral gap: ", spectral_gap)
    if PRINT_INFO: print("Condition number: ", condition_number)
    if PRINT_INFO: print("Dominance ratio: ", dominance_ratio)

    return eigenvalues

def plot_hist(pdf, title):
    # Sort the dictionary by the keys
    pdf = dict(sorted(pdf.items()))
    plt.bar(pdf.keys(), pdf.values())
    plt.xlabel("Letters")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution of the " + title)
    plt.show()

def plot_joint_hist_3D(joint_pmf, title):

    # sort the dictionary by the alphabet
    joint_pmf = dict(sorted(joint_pmf.items()))
    keys = list(joint_pmf.keys())
    joint_pmf = np.array(list(joint_pmf.values()))
    
    # Plot the joint pdf of the message and the cypher in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(26)
    y = np.arange(26)
    XX, YY = np.meshgrid(x, y)
    X = XX.ravel()
    Y = YY.ravel()
    Z = joint_pmf.ravel()
    bottom = np.zeros_like(Z)
    width = depth = 1

    values = np.linspace(0.2, 1., X.shape[0])
    colors = cm.Blues(values)
    ax.bar3d(X, Y, bottom, width, depth, Z, color=colors)
    ax.set_xticks(np.arange(26))
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_yticks(np.arange(26))
    ax.set_yticklabels(keys, fontsize=10)
    ax.set_xlabel("Message")
    ax.set_ylabel("Cypher")
    ax.set_zlabel("Frequency")
    ax.set_title("Joint Probability Distribution of " + title)
    plt.tight_layout()
    plt.show()

def plot_eigenvalues(eigenvalues):
    plt.plot(eigenvalues)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Frequency")
    plt.title("Eigenvalues of the Joint pmf of dependent Message and Cypher")#, msgLen = " + str(size))
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def main():
    print("One-Time Pad Encryption Algorithm")
    message, cypher, key = OneTimePadEncryptionAlgorithm()
    msgLen = len(message)
    if PRINT_INFO: print("The length of the message is: ", msgLen)

    print("Entropy calculations")
    entropy_calculations(msgLen)# Calculate the entropies of guessing one character of the key

    # Calculate and plot the probability distribution of the message
    pdf_msg = probability_distribution(message)
    if SHOW_PLOTS: plot_hist(pdf_msg, "Message")

    # And plot the pdf of the cypher
    pdf_cypher = probability_distribution(cypher)
    if SHOW_PLOTS: plot_hist(pdf_cypher, "Cypher")

    # and plot the pdf of the key
    pdf_key = probability_distribution(key)
    if SHOW_PLOTS: plot_hist(pdf_key, "Key")
    
    # Calculate and plot the joint pdf of the message and the cypher given the message if they are independent
    joint_pmf_indep = calculate_joint_independent_pmf(pdf_msg, pdf_cypher)
    if SHOW_PLOTS: plot_joint_hist_3D(joint_pmf_indep, "independent Message and Cypher")
    
    # Calculate and plot the joint pdf of the message and the cypher given the message if they are dependent
    joint_pmf_dep = calculate_joint_dependent_pmf(message, cypher, pdf_msg.keys())
    if SHOW_PLOTS: plot_joint_hist_3D(joint_pmf_dep, "dependent Message and Cypher")

    # Calculate, sort and plot the eigenvalues of the joint pmf
    eigenvalues = calculate_eigenvalues(joint_pmf_dep)
    if SHOW_PLOTS: plot_eigenvalues(eigenvalues)

    # Now calculate the pdf's of the cypher probabilities by 1000 different keys, randomly generated
    pdf_cypher_1000 = {}
    for i in range(1000):
        key = generate_key(msgLen)
        cypher = encrypt(message, key)
        pdf_cypher_1000 += probability_distribution(cypher)
    # Normalize the pdf_cypher
    for i in pdf_cypher_1000:
        pdf_cypher_1000[i] = pdf_cypher_1000[i] / len(1000)
    
    #plot a histogram
    plot_hist(pdf_cypher_1000)

if __name__ == "__main__":
    main()
