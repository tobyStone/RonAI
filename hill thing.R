# Investigate the hill heights

library(readr)
hill_heights <- read_csv("hill_heights.csv")


plot(x = hill_heights$longitude, y = hill_heights$latitude, type = "l", col = "black", lwd = 3,
     xlab = "Longitude", ylab = "Latitude")

library(leaflet)

# Create a leaflet map
m <- leaflet()

# Add map tiles
m <- addProviderTiles(m, "OpenStreetMap.Mapnik")

# Add markers for each point
for (i in 1:nrow(hill_heights)) {
  m <- addMarkers(m, lng = hill_heights$longitude[i], lat = hill_heights$latitude[i])
}

# Display the map
m

# Load required libraries
library(plotly)

# Set Mapbox access token
mapbox_token <- Sys.getenv("MAPBOX_ACCESS_TOKEN")  # Replace with your Mapbox access token

# Determine the center and zoom level based on the data
center_lat <- mean(hill_heights$latitude)
center_lon <- mean(hill_heights$longitude)

# Create a scattermapbox plot
plot <- plot_ly(hill_heights, type = "scattermapbox", mode = "markers") %>%
  add_trace(
    lat = ~latitude,
    lon = ~longitude,
    text = ~name
  ) %>%
  layout(
    mapbox = list(
      style = "satellite",
      zoom = 17,
      center = list(lon = center_lon, lat = center_lat),
      accesstoken = "pk.eyJ1Ijoiam9lamNvbGxpbnMiLCJhIjoiY2xqbzd3bXd4MTV4djNucjc4MTVxM2IwbyJ9.BdChwVAfLRJ8FDwZDaT_6g"
    ),
    margin = list(l = 0, r = 0, t = 0, b = 0)
  )

# Display the plot
plot
