
about_me = {
    'description': (
        "I am a Computer Science graduate from The George Washington University with a strong foundation in software engineering, distributed systems, and machine learning.\n\n"
        "Throughout my academic and professional journey, I have worked on projects ranging from real-time object detection using distributed systems to developing scalable APIs, real-time chat features, and cloud-native applications.\n\n"
        "Previously, I gained hands-on experience as a Software Engineer at Wavelet, where I built RESTful APIs with Python Flask, optimized MongoDB queries, and integrated real-time chat capabilities.\n\n"
        "I also explored game development and network security through internships, strengthening my ability to adapt across different domains.\n\n"
        "My technical skill set spans Python, Java, JavaScript, C++, Flask, Node.js, Angular, and AWS, and I enjoy building solutions that combine efficiency with user impact.\n\n"
        "Beyond academics, I enjoy badminton, exploring new places, and organizing food drives to give back to the community.\n\n"
        "I am passionate about building scalable, high-performance systems and excited to contribute my skills to impactful engineering challenges."
    )
}

experience = [
    {
        "company": "Wavelet",
        "position": "Software Engineer II",
        "duration": "Jul 2021 – Jul 2023",
        "description": [
            "Developed and integrated real-time chat features that reduced context switching by 8%, enhancing user engagement by enabling in-app communication and minimizing reliance on third-party messaging platforms.",
            "Executed testing and debugging procedures to identify bottlenecks, improving reliability and reducing API latency by 50%.",
            "Designed scalable RESTful APIs using Flask and Java Spring, and leveraged AWS S3 for efficient storage, ensuring real-time chat updates and preventing duplication.",
            "Led Flask API migration to Java Spring Boot, redesigning architecture with design patterns for scalability and maintainability.",
            "Refactored the codebase for multi-cloud support, improving scalability and reducing redundancy.",
            "Led cloud infrastructure migration, transitioning clients to dedicated AWS environments and managing ECS, DocumentDB, and API Gateway configurations.",
            "Boosted server throughput by 50% under high load by implementing multithreading, reducing latency and improving API reliability.",
            "Collaborated with stakeholders to deliver product demos, conduct training sessions, and troubleshoot technical issues, ensuring smooth onboarding and adoption of new features across client environments."
        ],
        "skills": [
            "Flask", "Java Spring Boot", "RESTful APIs", "AWS ECS", "AWS S3",
            "Amazon DocumentDB", "API Gateway", "MongoDB", "Multithreading",
            "Scalability", "System Design"
        ],
        "logo": "assets/images/wavelet-logo.png"
    },
    {
        "company": "Poladrone",
        "position": "Software Engineer Intern",
        "duration": "Jan 2021 – Mar 2021",
        "description": [
            "Designed and coded an internal network desktop application for seamless data uploads to NAS data storage, improving data accessibility and workflow efficiency.",
            "Built custom QGIS plugins using PyQt and QGIS libraries, streamlining data preprocessing workflows.",
            "Ensured a smooth UI experience by implementing PyQt threads to prevent GUI freezing during background tasks."
        ],
        "skills": ["Python", "PyQt", "QGIS", "NAS"],
        "logo": "assets/images/poladrone-logo.png"
    },
    {
        "company": "Gunfire Game Studio",
        "position": "Game Developer Intern",
        "duration": "Sep 2020 – Nov 2020",
        "description": [
            "Contributed to the development of FPS games using Unity and C#, focusing on gameplay mechanics and testing, which enhanced game performance and player engagement.",
            "Developed a First Person Shooting Controller (FPSC) for an engaging hunting game, improving player control and game realism."
        ],
        "skills": ["Unity", "C#", "Game Development"],
        "logo": "assets/images/gunfire-logo.png"
    }
]

projects = [
    {
        "id": 1,
        "title": "Real-time Object Detection using Distributed Systems",
        "description": [
            "Led a team of 3 to build a robust real-time object detection system using distributed systems.",
            "Utilized YOLOv8 as the base model, deployed in a hierarchical manner for efficiency, achieving a 1.5x speedup in image processing compared to baseline YOLOv8 large.",
            "Enhanced fault tolerance and scalability with load balancers and auto-scaling to maintain consistent performance under varying traffic conditions."
        ],
        "thumbnail": "assets/images/projects/object-detection.png",
        "technologies": ["Python", "YOLOv8", "Distributed Systems", "Load Balancing", "Auto-scaling"],
        "githubUrl": "https://github.com/yourusername/object-detection-distributed",
        "demoUrl": "",
        "featured": True,
        "category": "Machine Learning"
    },
    {
        "id": 2,
        "title": "AI Traffic Control",
        "description": [
            "Implemented a reinforcement learning agent to optimize traffic flow at intersections using Q-Learning and the SUMO traffic simulation environment.",
            "Developed and trained the agent to dynamically manage intersection traffic, reducing congestion and improving efficiency."
        ],
        "thumbnail": "assets/images/projects/ai-traffic.png",
        "technologies": ["Python", "Reinforcement Learning", "Q-Learning", "SUMO"],
        "githubUrl": "https://github.com/yourusername/ai-traffic-control",
        "demoUrl": "",
        "featured": False,
        "category": "AI"
    },
    {
        "id": 3,
        "title": "Final Year Deepfake Project",
        "description": [
            "Led a team of 3 to develop a novel GAN variation for deepfake detection and generation.",
            "Enhanced posture and facial expression handling within an existing GAN framework, improving realism and detection accuracy."
        ],
        "thumbnail": "assets/images/projects/deepfake.png",
        "technologies": ["Python", "GANs", "Deep Learning", "Computer Vision"],
        "githubUrl": "https://github.com/yourusername/deepfake-project",
        "demoUrl": "",
        "featured": False,
        "category": "AI"
    },
    {
        "id": 4,
        "title": "CupShup Rooftop App",
        "description": [
            "Developed an Android application for a restaurant in Lahore, Pakistan.",
            "Built with Java and Android Studio, integrated Firebase for order management.",
            "Improved customer satisfaction and sales through a streamlined order process and user-friendly interface."
        ],
        "thumbnail": "assets/images/projects/cupshup.png",
        "technologies": ["Java", "Android Studio", "Firebase"],
        "githubUrl": "https://github.com/yourusername/cupshup-rooftop-app",
        "demoUrl": "",
        "featured": False,
        "category": "Mobile"
    }
]


skills = {
    "programming_languages": ["Python", "Java", "C++", "JavaScript"],
    "frameworks": ["Flask", "Node.js", "Angular", "Java Spring Boot"],
    "cloud": ["AWS ECS", "S3", "DocumentDB", "API Gateway"],
    "tools": ["MongoDB", "QGIS", "Unity", "PyQt", "Git", "Docker"],
    "concepts": ["Distributed Systems", "Machine Learning", "Reinforcement Learning", "System Design"]
}
education = [
    {
        "degree": "Master of Science in Computer Science",
        "university": "The George Washington University",
        "graduation_year": 2025,
        "focus": ["Distributed Systems", "Machine Learning"]
    },
    {
        "degree": "Bachelor of Computer Science",
        "university": "Monash University, Malaysia",
        "minor": "Network and Cyber Security",
        "achievements": ["High distinction in Deepfake project", "Research on AI post-COVID opportunities"]
    }
]
faqs = [
    {
        "question": "What projects have you worked on?",
        "answer": "I have worked on projects including real-time object detection using distributed systems, AI traffic control using reinforcement learning, a deepfake GAN project, and Android apps like CupShup Rooftop."
    },
    {
        "question": "What programming languages do you know?",
        "answer": "I am proficient in Python, Java, C++, and JavaScript."
    },
    {
        "question": "What kind of roles are you interested in?",
        "answer": "I am passionate about software engineering roles that involve building scalable, high-performance systems, particularly in cloud computing, distributed systems, and AI."
    }
]
contacts = {
    "email": "hamzabashir1022@gmail.com",
    "linkedin": "https://www.linkedin.com/in/hamzabashir1022",
    "github": "https://github.com/yourusername",
    "portfolio": "https://yourportfolio.com"
}
personality = {
    "tone": "friendly, professional, enthusiastic",
    "greetings": ["Hi there!", "Hello! How can I help you today?", "Hey! Nice to meet you!"],
    "farewells": ["Goodbye!", "Have a great day!", "Talk to you later!"]
}

def get_projects():
    return projects

def get_experience():
    return experience

def get_about_me():
    return about_me
