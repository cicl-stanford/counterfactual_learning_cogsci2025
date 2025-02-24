function UUID() {
  const baseName =
    Math.floor(Math.random() * 10) +
    "" +
    Math.floor(Math.random() * 10) +
    "" +
    Math.floor(Math.random() * 10) +
    "" +
    Math.floor(Math.random() * 10);
  const template = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";
  const id =
    baseName +
    "-" +
    template.replace(/[xy]/g, function (c) {
      let r = (Math.random() * 16) | 0,
        v = c == "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  return id;
}

function randSign() {
  return Math.random() < 0.5 ? -1 : 1
}
function convertToHSL(color) {
  return `hsl(${color.h}, ${color.s}%, ${color.l}%)`;
}

// draws a rounded rectangle on the canvas
function drawRoundedRect(ctx, x, y, width, height, radius, fillStyle) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();

  // Set the fill style if provided and fill the rectangle
  if (fillStyle) {
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
}

// draws a roadblock sign on the canvas
function drawRoadblock(ctx, x, y, width, height, lineWidth, strokeStyle) {
  ctx.fillStyle = strokeStyle;
  ctx.beginPath();
  ctx.arc(x + width / 2, y + height / 2, width / 2, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "white";
  ctx.fillRect(x + width / 2 - width / 3, y + height / 2 - height / 8, width / 1.5, height / 4);
}

// defined parametrization for hue in HSL coordinates
function interpolateHue(baseHue, pathHue, t) {
  let distance;
  if (Math.abs(pathHue - baseHue) <= 180) {
    // The direct path is the shortest
    distance = pathHue - baseHue;
  } else {
    // We need to go around the hue circle
    if (pathHue > baseHue) {
      distance = pathHue - baseHue - 360;
    } else {
      distance = pathHue - baseHue + 360;
    }
  }
  let interpolatedHue = baseHue + distance * t;
  // Ensure the interpolated hue is within the 0-359 range
  interpolatedHue = (interpolatedHue + 360) % 360;
  return interpolatedHue;
}

// interpolates between two colors in HSL space
function interpolateColor(color1, color2, t) {
  const result = {
    h: interpolateHue(color1.h, color2.h, t),
    s: color1.s + (color2.s - color1.s) * t,
    l: color1.l + (color2.l - color1.l) * t,
  };
  return result;
}

// finds the closest distance between a point and an array of points
function findClosestDistance(points, point) {
  let minDistance = Infinity;
  points.forEach((p) => {
    const d = Math.sqrt(
      Math.pow(p.offsetX - point.offsetX, 2) +
        Math.pow(p.offsetY - point.offsetY, 2)
    );
    if (d < minDistance) {
      minDistance = d;
    }
  });
  return minDistance;
}

// returns a range of values evenly spaced between a start and end value
function makeRangeArr(startValue, stopValue, cardinality) {
  let arr = [];
  const step = (stopValue - startValue) / (cardinality - 1);
  for (let i = 0; i < cardinality; i++) {
    arr.push(startValue + step * i);
  }
  return arr;
}

function findCommonElements(coordsA, coordsB) {
  let commonElements = [];
  coordsA.forEach(function (elem1) {
    coordsB.forEach(function (elem2) {
      if (elem1[0] === elem2[0] && elem1[1] === elem2[1]) {
        commonElements.push(elem1);
      }
    });
  });
  return commonElements;
}

function findStartPosition(states) {
  for (let key in states) {
    if (states[key] === "start") {
      const coords = key.match(/\d+/g); // Extract digits from the key
      return { x: parseInt(coords[0]), y: parseInt(coords[1]) };
    }
  }
  return { x: 0, y: 0 };
}

function forceFullscreen() {
  if (document.fullscreenElement == null) {
    const fullScreen_div = document.createElement("div");
    const fullScreen_text = document.createElement("P");
    const fullScreen_button = document.createElement("button");
    fullScreen_div.id = "fullScreenPrompt";
    fullScreen_text.id = "fullScreenText";
    fullScreen_button.id = "fullScreenButton";
    fullScreen_text.textContent =
      "Please click the button to enter fullscreen.";
    fullScreen_button.textContent = "Enter fullscreen";

    fullScreen_button.onclick = function () {
      fullScreen_div.remove();
      document.documentElement.requestFullscreen();
    };

    $("body").append(fullScreen_div);
    fullScreen_div.appendChild(fullScreen_button);
    fullScreen_div.appendChild(fullScreen_text);
  }
}

function isValidMove(x, y, gridWidth, gridHeight, walls, visited) {
  return (
    x >= 0 &&
    x < gridWidth &&
    y >= 0 &&
    y < gridHeight &&
    !walls.some((wall) => wall.x === x && wall.y === y) &&
    !visited[y][x]
  );
}

function pathExists(startPos, goalPos, walls, gridWidth, gridHeight) {
  const queue = [startPos];
  const visited = Array.from({ length: gridHeight }, () => Array(gridWidth).fill(false));
  visited[startPos[1]][startPos[0]] = true;

  const directions = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
  ];

  while (queue.length > 0) {
    const [x, y] = queue.shift();

    if (x === goalPos[0] && y === goalPos[1]) {
      return true;
    }

    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      if (isValidMove(nx, ny, gridWidth, gridHeight, walls, visited)) {
        visited[ny][nx] = true;
        queue.push([nx, ny]);
      }
    }
  }

  return false;
}