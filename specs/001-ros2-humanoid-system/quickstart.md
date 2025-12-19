# Quickstart: ROS 2 for Humanoid Robotics

## Prerequisites
- Node.js 18+ and npm
- Git
- Basic knowledge of command line
- ROS 2 Humble Hawksbill or later (for testing examples)

## Setup Instructions

### 1. Clone and Initialize the Documentation
```bash
git clone [repository-url]
cd [repository-name]
npm install
```

### 2. Start Development Server
```bash
npm start
```
This will start the Docusaurus development server and open your documentation in the browser at http://localhost:3000

### 3. Build for Production
```bash
npm run build
```
This creates an optimized production build in the `build/` directory

### 4. Add New Content
- Create new Markdown files in the `docs/` directory
- Update `sidebar.js` to add navigation links
- Use Docusaurus frontmatter to specify page metadata

## Running ROS 2 Examples
The examples in this documentation use Python and rclpy. To run them:

1. Source your ROS 2 installation:
```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
```

2. Create a Python virtual environment:
```bash
python3 -m venv ros2_examples_env
source ros2_examples_env/bin/activate
```

3. Run the examples as provided in each chapter

## Documentation Structure
- `docs/modules/ros2-humanoid-system/` - Contains the three main chapters
- `docusaurus.config.js` - Main configuration file
- `sidebar.js` - Navigation structure
- `src/components/` - Custom React components
- `static/` - Static assets like images

## Deployment
The documentation is designed to be deployed to GitHub Pages. The build process creates a static site that can be served from any web server.