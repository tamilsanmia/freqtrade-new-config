#!/bin/bash

echo "Docker Cleanup Script"
echo "----------------------"

# Show current Docker disk usage
echo -e "\nCurrent Docker disk usage:"
docker system df

# Prompt for basic cleanup
read -p $'\nDo you want to run basic cleanup (remove stopped containers, dangling images, unused networks)? [y/N]: ' basic
if [[ "$basic" =~ ^[Yy]$ ]]; then
    docker system prune -f
fi

# Prompt for volume cleanup
read -p $'\nDo you also want to remove unused volumes? [y/N]: ' vol
if [[ "$vol" =~ ^[Yy]$ ]]; then
    docker system prune --volumes -f
fi

# Prompt for full cleanup
read -p $'\nDo you want to remove ALL unused images (not just dangling)? [y/N]: ' all_images
if [[ "$all_images" =~ ^[Yy]$ ]]; then
    docker image prune -a -f
fi

# Final disk usage
echo -e "\nFinal Docker disk usage:"
docker system df

echo -e "\n✅ Docker cleanup completed."
