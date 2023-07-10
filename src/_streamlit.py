import streamlit as st
import requests
from PIL import Image
import util as util

config = util.load_config()

# Load and set images in the first place
header_images = Image.open('pics/PPSK_2022_TMEC_Header-Image_CAC-2.png')
st.image(header_images)

# Add some information about the service
st.title("Customer Cost Acquisition Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "Customer_Cost_Acquistion_Form"):
    # Create Box For Number Input
    store_cost = st.number_input(
        label = "Enter store_cost Value:",
        min_value = config["store_cost"][0],
        max_value = config["store_cost"][1],
        help = "Value range from 1632163.0 to 97274727.0"
    )
    
    total_children = st.number_input(
        label = "Enter total_children Value:",
        min_value = config["total_children"][0],
        max_value = config["total_children"][1],
        help = "Value range from 0 to 5"
    )
    
    avg_cars_at_home = st.number_input(
        label = "Enter avg_cars_at_home Value:",
        min_value = config["avg_cars_at_home"][0],
        max_value = config["avg_cars_at_home"][1],
        help = "Value range from 0 to 4"
    )
    
    num_children_at_home = st.number_input(
        label = "Enter num_children_at_home Value:",
        min_value = config["num_children_at_home"][0],
        max_value = config["num_children_at_home"][1],
        help = "Value range from 0 to 5"
    )
    
    net_weight = st.number_input(
        label = "Enter net_weight Value:",
        min_value = config["net_weight"][0],
        max_value = config["net_weight"][1],
        help = "Value range from 3 to 21"
    )
    
    units_per_case = st.number_input(
        label = "Enter units_per_case Value:",
        min_value = config["units_per_case"][0],
        max_value = config["units_per_case"][1],
        help = "Value range from 1 to 36"
    )
    
    coffee_bar = st.number_input(
        label = "Enter coffee_bar Value:",
        min_value = config["coffee_bar"][0],
        max_value = config["coffee_bar"][1],
        help = "Value range from 0 to 1"
    )
    
    video_store = st.number_input(
        label = "Enter video_store Value:",
        min_value = config["video_store"][0],
        max_value = config["video_store"][1],
        help = "Value range from 0 to 1"
    )
    
    prepared_food = st.number_input(
        label = "Enter prepared_food Value:",
        min_value = config["prepared_food"][0],
        max_value = config["prepared_food"][1],
        help = "Value range from 0 to 1"
    )
    
    florist = st.number_input(
        label = "Enter florist Value:",
        min_value = config["florist"][0],
        max_value = config["florist"][1],
        help = "Value range from 0 to 1"
    )
    
    # Create select box input
    promotion_name = st.selectbox(
        label = "From which promotion_name is this data collected?",
        options = (
            "Cash Register Lottery",
            "Wallet Savers",
            "One Day Sale",
            "Bye Bye Baby",
            "Savings Galore",
            "Saving Days",
            "Save-It Sale",
            "Super Duper Savers",
            "Price Winners",
            "Two for One",
            "Price Savers",
            "Money Savers",
            "Free For All",
            "Big Time Discounts",
            "Dollar Cutters",
            "High Roller Savings",
            "Price Cutters",
            "Weekend Markdown",
            "Super Savers",
            "Green Light Days",
            "Shelf Emptiers",
            "I Cant Believe It Sale",
            "Sales Days",
            "Two Day Sale",
            "Sale Winners",
            "Double Down Sale",
            "Price Slashers",
            "Big Time Savings",
            "Three for One",
            "Green Light Special",
            "Coupon Spectacular",
            "Big Promo",
            "Bag Stuffers",
            "Tip Top Savings",
            "Shelf Clearing Days",
            "Go For It",
            "Unbeatable Price Savers",
            "Super Wallet Savers",
            "Dimes Off",
            "Price Smashers",
            "Pick Your Savings",
            "Sales Galore",
            "Price Destroyers",
            "Dollar Days",
            "You Save Days",
            "Best Savings",
            "Mystery Sale",
            "Fantastic Discounts",
            "Double Your Savings"
        )
    )
    
    sales_country = st.selectbox(
        label = "From which sales_country is this data collected?",
        options = (
            "USA",
            "Mexico", 
            "Canada"
        )
    )
    
    occupation = st.selectbox(
        label = "From which occupation is this data collected?",
        options = (
            "Professional",
            "Manual",
            "Clerical",
            "Management",
            "Skilled Manual"
        )
    )
    
    avg_yearly_income = st.selectbox(
        label = "From which avg_yearly_income is this data collected?",
        options = (
            "$30K - $50K",
            "$10K - $30K",
            "$50K - $70K",
            "$70K - $90K",
            "$130K - $150K",
            "$90K - $110K",
            "$110K - $130K",
            "$150K +"
        )
    )
    
    store_type = st.selectbox(
        label = "From which store_type is this data collected?",
        options = (
            "Deluxe Supermarket",
            "Supermarket",
            "Small Grocery",
            "Gourmet Supermarket",
            "Mid-Size Grocery"
        )
    )
    
    store_city = st.selectbox(
        label = "From which store_city is this data collected?",
        options = (
            "Salem",
            "Orizaba",
            "Bellingham",
            "Beverly Hills",
            "Portland",
            "Spokane",
            "Camacho",
            "San Francisco",
            "Bremerton",
            "Tacoma",
            "Seattle",
            "Mexico City",
            "Merida",
            "Los Angeles",
            "Guadalajara",
            "Vancouver",
            "Hidalgo",
            "Victoria",
            "Acapulco"  
        )
    )
    
    store_state = st.selectbox(
        label = "From which store_state is this data collected?",
        options = (
            "OR",
            "Veracruz",
            "WA",
            "CA",
            "Zacatecas",
            "DF",
            "Yucatan",
            "Jalisco",
            "BC",
            "Guerrero"   
        )
    )
    
    media_type = st.selectbox(
        label = "From which media_type is this data collected?",
        options = (
            "Sunday Paper, Radio",
            "TV",
            "Cash Register Handout",
            "Daily Paper, Radio, TV",
            "Product Attachment",
            "Daily Paper, Radio",
            "Daily Paper",
            "Sunday Paper, Radio, TV",
            "Street Handout",
            "In-Store Coupon",
            "Sunday Paper",
            "Bulk Mail",
            "Radio"
        )
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "store_cost": store_cost,
            "total_children": total_children,
            "avg_cars_at_home": avg_cars_at_home,
            "num_children_at_home": num_children_at_home,
            "net_weight": net_weight,
            "units_per_case": units_per_case,
            "coffee_bar": coffee_bar,
            "video_store": video_store,
            "prepared_food": prepared_food,
            "florist": florist,
            "promotion_name": promotion_name,
            "sales_country": sales_country,
            "occupation": occupation,
            "avg_yearly_income": avg_yearly_income,
            "store_type": store_type,
            "store_city": store_city,
            "store_state": store_state,
            "media_type": media_type
        }
        
        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()
        
        # print(res)

        # Parse the prediction result
        if "error_msg" in res:
            if res["error_msg"] != "":
                st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
            else:
                # Process the prediction result
                result = res["res"]
                # Display the result in Streamlit
                st.write(result)
        else:
            st.error("Invalid response from the prediction server")
    
    
    
    
    


