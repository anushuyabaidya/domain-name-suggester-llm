"""
Created By: Anushuya Baidya
Date: 7/28/25
"""
import random
import pandas as pd

class BusinessDescriptionGenerator:
    """Generates realistic business descriptions for training"""
    def __init__(self):
        self.business_data = {
            "tech": {
                "types": ["app development", "software company", "startup", "SaaS platform"],
                "specialties": ["AI", "mobile apps", "web development", "automation"],
                "style": ["innovative", "cutting-edge", "fast-growing", "scalable"]
            },
            "food": {
                "types": ["coffee shop", "restaurant", "bakery", "food truck"],
                "specialties": ["organic", "vegan", "artisan", "farm-to-table"],
                "style": ["local", "family-owned", "cozy", "fresh"]
            },
            "health": {
                "types": ["yoga studio", "fitness center", "spa", "wellness clinic"],
                "specialties": ["holistic", "therapeutic", "relaxing", "healing"],
                "style": ["peaceful", "professional", "welcoming", "modern"]
            },
            "retail": {
                "types": ["boutique", "online store", "shop", "marketplace"],
                "specialties": ["handmade", "vintage", "designer", "sustainable"],
                "style": ["trendy", "affordable", "unique", "quality"]
            }
        }

        self.templates = [
            "{style} {type} specializing in {specialty}",
            "{specialty} {type} in the {style} district",
            "We run a {style} {type} focused on {specialty}",
            "Local {specialty} {type} with {style} approach",
            "{type} offering {specialty} services"
        ]

    def create_one_description(self, category=None):
        """Create one business description"""
        if not category:
            category = random.choice(list(self.business_data.keys()))
        data = self.business_data[category]
        business_type = random.choice(data["types"])
        specialty = random.choice(data["specialties"])
        style = random.choice(data["style"])
        template = random.choice(self.templates)
        description = template.format(
            type=business_type,
            specialty=specialty,
            style=style
        )

        return {
            "description": description,
            "category": category,
            "business_type": business_type,
            "specialty": specialty,
            "style": style
        }

    def create_training_data(self, total_samples=400):
        """Create training dataset"""
        all_data = []
        samples_per_category = total_samples // len(self.business_data)
        for category in self.business_data.keys():
            for _ in range(samples_per_category):
                sample = self.create_one_description(category)
                all_data.append(sample)
        return pd.DataFrame(all_data)

class DomainGenerator:
    """Generates domain suggestions from business descriptions"""
    def __init__(self):
        self.extensions = [".com", ".net", ".org", ".io"]

    def extract_keywords(self, business_data):
        """Extract keywords from business description parts"""
        keywords = []
        type_words = business_data["business_type"].replace(" ", "").lower()
        keywords.append(type_words)

        specialty = business_data["specialty"].lower()
        if " " not in specialty:
            keywords.append(specialty)

        style = business_data["style"].lower()
        if style in ["local", "smart", "quick", "best", "top"]:
            keywords.append(style)
        return keywords

    def create_domains(self, business_data):
        """Create domain suggestions"""
        keywords = self.extract_keywords(business_data)
        domains = []

        # Pattern 1: single keyword + extension
        if keywords:
            domain = keywords[0] + random.choice(self.extensions)
            domains.append(domain)

        # Pattern 2: combine two keywords
        if len(keywords) >= 2:
            domain = keywords[0] + keywords[1] + random.choice(self.extensions)
            domains.append(domain)

        # Pattern 3: keyword + "hub"/"pro"/"co"
        if keywords:
            suffix = random.choice(["hub", "pro", "co", "spot"])
            domain = keywords[0] + suffix + random.choice(self.extensions)
            domains.append(domain)

        return domains[:3]  # Return max 3 domains

# Complete Training Data Creator
def create_complete_training_data(num_samples=400):
    """Create complete training data with descriptions and domains"""
    print(f"Creating {num_samples} training samples...")

    desc_generator = BusinessDescriptionGenerator()
    business_df = desc_generator.create_training_data(num_samples)
    domain_generator = DomainGenerator()
    training_data = []
    for _, row in business_df.iterrows():
        business_data = {
            "business_type": row["business_type"],
            "specialty": row["specialty"],
            "style": row["style"]
        }

        domains = domain_generator.create_domains(business_data)
        training_text = f"Business: {row['description']} Domains: {', '.join(domains)}"
        training_data.append({
            "business_description": row["description"],
            "category": row["category"],
            "domains": domains,
            "training_text": training_text
        })

    training_df = pd.DataFrame(training_data)
    print(f"Created {len(training_df)} training samples")
    print(f"Categories: {training_df['category'].value_counts().to_dict()}")
    return training_df

if __name__ == "__main__":
    training_file_path = "../data/training_data.csv"
    generator = BusinessDescriptionGenerator()
    print("Creating Complete Training Data")
    training_data = create_complete_training_data(1000)
    print("\nTraining Examples:")
    for i in range(3):
        print(f"{i+1}. {training_data.iloc[i]['training_text']}")

    training_data.to_csv(training_file_path, index=False)
    print(f"\nSaved training data to training_data.csv")
