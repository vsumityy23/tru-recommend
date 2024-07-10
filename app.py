
import streamlit as st
import pandas as pd

#####################################################  STREAMLIT UI ##################################################################################
# Set the page config
st.set_page_config(page_title="TRU Recommend", layout="wide")

# Title and mode switch
col1, col2 = st.columns([3, 1])
with col1:
    st.title("TRU Recommend")
with col2:
    mode = st.radio("Select Mode", ('User Mode','Client Mode'))

st.write(f"You are in {mode}")

# Main section based on mode
if mode == 'User Mode':
    
    
    st.subheader("Get your personalized projects")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    st.write('or fill manually')
    with st.spinner("Loading..."):    
        if uploaded_file is not None:
            
            from resumeextract import extract_resume_info
            # Process the uploaded PDF
            pdf_content = uploaded_file.read()
            extraction_result = extract_resume_info(pdf_content)
            skill_pre=extraction_result["skills"]
            tools_pre=extraction_result["tools"]
            certifications_pre=extraction_result["certifications"]
            
            st.session_state['skill'] = skill_pre
            st.session_state['tool'] = tools_pre
            st.session_state['cer'] = certifications_pre
            
        from utility import skills,tools,certifications,projects
        
        
        skills = st.multiselect("Select Skills", skills,key='skill')
        tools = st.multiselect("Select Tools", tools,key='tool')
        certifications = st.multiselect("Select Certifications",certifications,key='cer')
        # past_projects = st.multiselect("Select Past Projects", projects)
        past_projects_str= st.text_input("Past Projects",placeholder="Enter your past projects..")
        print(past_projects_str)
        past_projects=[past_projects_str]
        model_name = st.radio("Select Model", ('Bert', 'GPT2', 'Roberta'), index=0)
    

            
        
    if st.button("Recommend"):
        
        
        st.subheader("Recommended projects for you:")
    
        with st.spinner("Loading...."):
        #load models as per requirement
            if (model_name=='Bert'):
                from utility import bert_model,bert_tokenizer
                model=bert_model
                tokenizer=bert_tokenizer
                embedding_name='bert_Embeddings'
                
            
            elif(model_name=='GPT2'):
                from utility import gpt2_tokenizer,gpt2_model
                
                model=gpt2_model
                tokenizer=gpt2_tokenizer
                embedding_name='gpt2_Embeddings'
                    
            elif(model_name=='Roberta'):
                from utility import roberta_model,roberta_tokenizer
                
                tokenizer=roberta_tokenizer
                model=roberta_model
                embedding_name='roberta_Embeddings'
            
            from utility import get_embeddings,recommend_projects
            projects_df = pd.read_pickle('./embedding_pickelfile/projects_with_embeddings.pkl')            
            input_text = ' '.join(skills + tools + certifications + past_projects)
            input_embedding=get_embeddings(input_text,tokenizer,model)
            recommended_project=recommend_projects(input_embedding,projects_df,embedding_name, top_n=10)
            df=recommended_project[['Project_ID', 'Project_Name', 'Project_Description', 'Skills_Required']].reset_index(drop=True)
            st.table(df)

elif mode == 'Client Mode':
    
    st.subheader("Get the suitable students for your projects")
    with st.spinner('Loading....'):
        projects_name = pd.read_pickle('./embedding_pickelfile/projects_with_embeddings.pkl')['Project_Name'].tolist()
        selected_project = st.selectbox("Choose Project",projects_name,index=None,placeholder="Select Project...")
        project_str=st.text_area("Or ",placeholder="Write your project description...")
        
        if (selected_project==None):
            selected_project=project_str
        
        elif(project_str==None):
            selected_project=project_str
        else:
            selected_project=selected_project+project_str 
        
    model_name = st.radio("Select Model", ('Bert', 'GPT2', 'Roberta'), index=0)
        
    if st.button("Recommend"):
        
        st.write("Recommended students for your projects:")
        with st.spinner('Loading...'):
            resumes_df = pd.read_pickle('./embedding_pickelfile/resumes_with_embeddings.pkl')
            
            #load models as per requirement
            if (model_name=='Bert'):
                from utility import bert_model,bert_tokenizer
                model=bert_model
                tokenizer=bert_tokenizer
                embedding_name='bert_Embeddings'
                
            
            elif(model_name=='GPT2'):
                from utility import gpt2_tokenizer,gpt2_model
                
                model=gpt2_model
                tokenizer=gpt2_tokenizer
                embedding_name='gpt2_Embeddings'
                    
            elif(model_name=='Roberta'):
                from utility import roberta_model,roberta_tokenizer
                
                tokenizer=roberta_tokenizer
                model=roberta_model
                embedding_name='roberta_Embeddings'
            
            from utility import get_embeddings,recommend_students
                
            input_embedding=get_embeddings(selected_project,tokenizer,model)
            recommended_project=recommend_students(input_embedding,resumes_df,embedding_name, top_n=10)
            df=recommended_project[['Student_ID','Skills','Tools','Certification','Past_Project_Descriptions']].reset_index(drop=True)
            st.table(df)
