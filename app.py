import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model, setup, get_config
from sklearn.preprocessing import LabelEncoder

def prepare_input_data(no_of_likes, no_of_images, title, item_condition, deal_method, post_date, category_type, post_type,
                       condition_subtext, mailing_option, meetup_option, meetup_location, seller_id, seller_join_date,
                       seller_response, seller_verif, verified_by_email, verified_by_facebook, verified_by_mobile,
                       seller_stars_rating, reviews_of_seller, matched_brand, current_listing_price,
                       len_posts, post_word_count, num_emojis, topic_probability, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    # Create a DataFrame with the user input
    data = pd.DataFrame({
        'no_of_likes': [no_of_likes],
        'no_of_images': [no_of_images],
        'title': [title],
        'item_condition': [item_condition],
        'deal_method': [deal_method],
        'post_date': [post_date],
        'category_type': [category_type],
        'post_type': [post_type],
        'condition_subtext': [condition_subtext],
        'mailing_option': [mailing_option],
        'meetup_option': [meetup_option],
        'meetup_location': [meetup_location],
        'seller_id': [seller_id],
        'seller_join_date': [seller_join_date],
        'seller_response': [seller_response],
        'seller_verif': [seller_verif],
        'verified_by_email': [verified_by_email],
        'verified_by_facebook': [verified_by_facebook],
        'verified_by_mobile': [verified_by_mobile],
        'seller_stars_rating': [seller_stars_rating],
        'reviews_of_seller': [reviews_of_seller],
        'matched_brand': [matched_brand],
        'current_listing_price': [current_listing_price],
        'len_posts': [len_posts],
        'post_word_count': [post_word_count],
        'num_emojis': [num_emojis],
        'topic_probability': [topic_probability],
        'x1': [x1],
        'x2': [x2],
        'x3': [x3],
        'x4': [x4],
        'x5': [x5],
        'x6': [x6],
        'x7': [x7],
        'x8': [x8],
        'x9': [x9],
        'x10': [x10]
    })
    return data

def label_encode_categorical(df):
    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()

    # Apply label encoding to each categorical column
    for column in df.columns:
        if df[column].dtype.name == 'category':
            df[column] = label_encoder.fit_transform(df[column])

    return df

def main():
    st.title("User Input Form")

    no_of_likes = st.number_input("Number of Likes", value=0, min_value=0)
    no_of_images = st.number_input("Number of Images", value=0, min_value=0)
    title = st.text_input("Title", value="Default Title")
    item_condition = st.selectbox("Item Condition", ["New", "Used", "Refurbished"])
    deal_method = st.selectbox("Deal Method", ["Shipping", "Meetup", "Both"])
    post_date = st.date_input("Post Date")
    category_type = st.text_input("Category Type", value="Default Category")
    post_type = st.selectbox("Post Type", ["For Sale", "For Trade", "For Auction"])
    condition_subtext = st.text_area("Condition Subtext", value="Default Subtext")
    mailing_option = st.checkbox("Mailing Option")
    meetup_option = st.checkbox("Meetup Option")
    meetup_location = st.text_input("Meetup Location", value="Default Location")
    seller_id = st.text_input("Seller ID", value="Default Seller ID")
    seller_join_date = st.date_input("Seller Join Date")
    seller_response = st.number_input("Seller Response", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
    seller_verif = st.checkbox("Seller Verified")
    verified_by_email = st.checkbox("Verified by Email")
    verified_by_facebook = st.checkbox("Verified by Facebook")
    verified_by_mobile = st.checkbox("Verified by Mobile")
    seller_stars_rating = st.slider("Seller Stars Rating", value=0, min_value=0, max_value=5, step=1)
    reviews_of_seller = st.number_input("Reviews of Seller", value=0, min_value=0, step=1)
    matched_brand = st.text_input("Matched Brand", value="Default Brand")
    current_listing_price = st.number_input("Current Listing Price", value=0.0, min_value=0.0, step=0.01)
    len_posts = st.number_input("Length of Posts", value=0, min_value=0)
    post_word_count = st.number_input("Post Word Count", value=0, min_value=0)
    num_emojis = st.number_input("Number of Emojis", value=0, min_value=0, step=1)
    topic_probability = st.slider("Topic Probability", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
    
    x1 = st.text_input("X1", value="Default X1")
    x2 = st.text_input("X2", value="Default X2")
    x3 = st.text_input("X3", value="Default X3")
    x4 = st.text_input("X4", value="Default X4")
    x5 = st.text_input("X5", value="Default X5")
    x6 = st.text_input("X6", value="Default X6")
    x7 = st.text_input("X7", value="Default X7")
    x8 = st.text_input("X8", value="Default X8")
    x9 = st.text_input("X9", value="Default X9")
    x10 = st.text_input("X10", value="Default X10")

    if st.button("Submit"):
        input_data = prepare_input_data(no_of_likes, no_of_images, title, item_condition, deal_method, post_date, category_type, post_type,
                                        condition_subtext, mailing_option, meetup_option, meetup_location, seller_id, seller_join_date,
                                        seller_response, seller_verif, verified_by_email, verified_by_facebook, verified_by_mobile,
                                        seller_stars_rating, reviews_of_seller, matched_brand, current_listing_price,
                                        len_posts, post_word_count, num_emojis, topic_probability, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)


        model = load_model('model')


        # Apply label encoding to the categorical columns
        input_data_encoded = label_encode_categorical(input_data)

        # Make predictions using the loaded model and the transformed input data
        predictions = model.predict(input_data_encoded)

        if predictions[0] == "LOW":
            st.write("Predictions:", "Recommended Price Range: Low (<=$700)")
        elif predictions[0] == "MEDIUM":
            st.write("Predictions:", "Recommended Price Range: Medium (<=$1500)")
        else:
            st.write("Predictions:", "Recommended Price Range: High (<=$2000)")            
if __name__ == "__main__":
    main()
