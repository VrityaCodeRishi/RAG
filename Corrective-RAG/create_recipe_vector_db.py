import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

embeddings = OpenAIEmbeddings()

urls = [
    "https://www.allrecipes.com/recipes/",
    "https://www.allrecipes.com/recipes/76/appetizers-and-snacks/",
    "https://www.allrecipes.com/recipes/78/breakfast-and-brunch/",
    "https://www.allrecipes.com/recipes/79/main-dish/",
    "https://www.allrecipes.com/recipes/80/desserts/",
    "https://www.allrecipes.com/recipes/156/bread/",
    "https://www.allrecipes.com/recipes/84/healthy-recipes/",
    "https://www.foodnetwork.com/recipes",
    "https://www.foodnetwork.com/recipes/photos/easy-dinner-recipes",
    "https://www.foodnetwork.com/recipes/photos/baking-recipes",
    "https://www.foodnetwork.com/how-to/packages/food-network-essentials",
    "https://www.bbcgoodfood.com/recipes",
    "https://www.bbcgoodfood.com/recipes/collection/easy-recipes",
    "https://www.bbcgoodfood.com/recipes/collection/quick-recipes",
    "https://www.bbcgoodfood.com/recipes/collection/healthy-recipes",
    "https://www.seriouseats.com/recipes",
    "https://www.seriouseats.com/cooking-techniques",
    "https://www.seriouseats.com/the-food-lab",
    "https://www.seriouseats.com/knife-skills",
    "https://www.thespruceeats.com/cooking-basics-4684041",
    "https://www.thespruceeats.com/cooking-methods-4684040",
    "https://www.allrecipes.com/article/cooking-basics/",
    "https://www.foodnetwork.com/how-to/packages/food-network-essentials/cooking-basics",
    "https://www.allrecipes.com/article/common-ingredient-substitutions/",
    "https://www.foodnetwork.com/how-to/packages/food-network-essentials/ingredient-substitutions",
    "https://www.thespruceeats.com/common-ingredient-substitutions-4684041",
    "https://www.bbcgoodfood.com/howto/guide/ingredient-substitutions",
    "https://www.allrecipes.com/recipes/1227/everyday-cooking/vegetarian/vegan/",
    "https://www.bbcgoodfood.com/recipes/collection/vegan-recipes",
    "https://www.foodnetwork.com/recipes/photos/vegan-recipes",
    "https://www.allrecipes.com/recipes/1227/everyday-cooking/vegetarian/gluten-free/",
    "https://www.bbcgoodfood.com/recipes/collection/gluten-free-recipes",
    "https://www.allrecipes.com/recipes/1227/everyday-cooking/healthy-recipes/low-carb/",
    "https://www.bbcgoodfood.com/recipes/collection/keto-recipes",
]

print("Loading documents from web...")
docs = []
for url in urls:
    try:
        print(f"  Loading: {url}")
        loader = WebBaseLoader(url)
        loaded_docs = loader.load()
        docs.extend(loaded_docs)
        print(f"    ✓ Loaded {len(loaded_docs)} document(s)")
    except Exception as e:
        print(f"    ✗ Error loading {url}: {e}")
        continue

print(f"\nTotal documents loaded: {len(docs)}")

print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(doc_splits)}")

print("\nCreating Chroma vector database...")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    collection_name="recipe_assistant",
    persist_directory="./chroma_db/recipes"
)

print("\n✓ Vector database created successfully!")
print(f"  - Location: ./chroma_db/recipes")
print(f"  - Total chunks: {len(doc_splits)}")
print(f"  - Collection name: recipe_assistant")

