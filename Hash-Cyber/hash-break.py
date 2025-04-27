import hashlib
import itertools
import string
import csv

# Step 1: Create a simple hash
original_text = "addy"  # You can change this to any short word
hash_object = hashlib.md5(original_text.encode())
hash_to_crack = hash_object.hexdigest()

print(f"Original text: {original_text}")
print(f"MD5 Hash: {hash_to_crack}")

# Step 2: Brute-force the hash and save logs to CSV
def brute_force_md5(target_hash, max_length=4):
    letters = string.ascii_lowercase  # a-z
    attempts = 0
    found_text = None

    # Open the CSV file to write
    with open('hash-crack.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Guess', 'Hash'])  # CSV Header

        for length in range(1, max_length + 1):
            for guess in itertools.product(letters, repeat=length):
                guess_word = ''.join(guess)
                guess_hash = hashlib.md5(guess_word.encode()).hexdigest()
                attempts += 1

                # Write every guess and its hash to CSV
                writer.writerow([guess_word, guess_hash])

                # Also print attempt
                #print(f"Trying: {guess_word} -> {guess_hash}")

                if guess_hash == target_hash:
                    found_text = guess_word
                    break
            if found_text:
                break

    return found_text, attempts

# Step 3: Perform brute force
found_text, total_attempts = brute_force_md5(hash_to_crack)

# Step 4: Print final summary
print("\n--- Summary ---")
print(f"Total attempts made: {total_attempts}")

if found_text:
    print(f"✅ Cracked Successfully! The text is: {found_text}")
else:
    print("❌ Failed to crack the hash.")
