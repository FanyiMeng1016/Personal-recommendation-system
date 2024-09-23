def recommend_magazine(gender, age):

    # Recommendation rules
    if gender == 'female':
        if 18 <= age <= 25:
            return "We recommend 'Dazed' for you. It's perfect for young women interested in cutting-edge fashion, art, and culture."
        elif 26 <= age <= 35:
            return "We recommend 'Cosmopolitan' for you. It's ideal for young women, covering fashion, lifestyle, and entertainment."
        elif 36 <= age <= 45:
            return "We recommend 'Elle' for you. It focuses on sophisticated fashion, beauty, and lifestyle content for mature women."
        elif age > 45:
            return "We recommend 'Vogue' for you. It's a timeless choice that covers high-end fashion, beauty, and culture, perfect for women who appreciate luxury and style."
        else:
            return "Your age is outside the recommended range. Please update your age for more accurate recommendations."
    
    elif gender == 'male':
        if 18 <= age <= 25:
            return "We recommend 'Dazed' for you. It's great for young men interested in avant-garde fashion, art, and culture."
        elif 26 <= age <= 35:
            return "We recommend 'GQ' for you. It covers men's fashion, lifestyle, and culture, perfect for young men."
        elif 36 <= age <= 45:
            return "We recommend 'Esquire' for you. It offers high-end fashion, lifestyle, and cultural content for mature men."
        elif age > 45:
            return "We recommend 'Esquire' for you. It's suitable for mature men who appreciate quality living and culture."
        else:
            return "Your age is outside the recommended range. Please update your age for more accurate recommendations."
    
    else:
        return "Invalid gender input. Please enter 'male' or 'female'."

# Get user input
user_gender = input("Please enter your gender (male or female): ").strip().lower()
user_age = int(input("Please enter your age: ").strip())

# Get recommendation
recommendation = recommend_magazine(user_gender, user_age)

# Print recommendation
print(recommendation)

