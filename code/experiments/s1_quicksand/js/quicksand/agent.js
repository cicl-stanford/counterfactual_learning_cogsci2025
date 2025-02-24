/**
 * Class representing an Agent in the Gridworld.
 * @param {number} x - The x-coordinate of the agent on the grid.
 * @param {number} y - The y-coordinate of the agent on the grid.
 * @param {number} cellSize - number of pixels in a grid cell.
 * @property {number} travelFrame - The current frame of the agent's travel animation.
 * @function draw - Draws the agent on the canvas.
 * @function setPosition - Sets the agent's position to the given coordinates.
 * @function moveAcrossTile - Moves the agent across the a tile in a supplied direction.
 */
class Agent {
  constructor(x, y, cellSize, color) {
    this.x = x;
    this.y = y;
    this.cellSize = cellSize;
    this.color = color;
    this.travelFrame = 0;
  }

  draw(ctx) {
    const pixelX = (this.x + 0.5) * this.cellSize;
    const pixelY = (this.y + 0.5) * this.cellSize;
    const radius = this.cellSize * gs.agent.size.radius;
    ctx.fillStyle = convertToHSL(this.color);
    ctx.beginPath();
    ctx.arc(pixelX, pixelY, radius, 0, Math.PI * 2);
    ctx.fill();

    /** Eyes */
    const eyeRadius = radius * gs.agent.size.eye_radius;
    const eyeOffsetX = radius / 3;
    const eyeOffsetY = radius / 3;
    ctx.fillStyle = "white";

    ctx.beginPath(); // Left eye
    ctx.arc(
      pixelX - eyeOffsetX,
      pixelY - eyeOffsetY,
      eyeRadius,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.beginPath(); // Right eye
    ctx.arc(
      pixelX + eyeOffsetX,
      pixelY - eyeOffsetY,
      eyeRadius,
      0,
      Math.PI * 2
    );
    ctx.fill();

    const pupilRadius = eyeRadius * gs.agent.size.pupil_radius;
    const pupilOffsetX = eyeRadius / 4;
    const pupilOffsetY = eyeRadius / 6;
    ctx.fillStyle = "black"; // Black part of the pupils

    ctx.beginPath(); // Left pupil
    ctx.arc(
      pixelX - eyeOffsetX + pupilOffsetX,
      pixelY - eyeOffsetY + pupilOffsetY,
      pupilRadius,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.beginPath(); // Right pupil
    ctx.arc(
      pixelX + eyeOffsetX + pupilOffsetX,
      pixelY - eyeOffsetY + pupilOffsetY,
      pupilRadius,
      0,
      Math.PI * 2
    );
    ctx.fill();

    /** Smile */
    const smileRadius = radius * gs.agent.size.smile_radius;
    const smileStartY = pixelY + radius / 5;
    const smileAngle = 0.0 * Math.PI; // Smaller angle for a "tighter" smile
    ctx.fillStyle = "white";
    ctx.beginPath(); // Smile
    ctx.arc(
      pixelX,
      smileStartY,
      smileRadius,
      smileAngle,
      Math.PI - smileAngle,
      false
    );
    ctx.lineTo(pixelX, smileStartY);
    ctx.closePath();
    ctx.fill();
  }

  setPosition(x, y) {
    this.x = x;
    this.y = y;
  }

  moveAcrossTile(tile, direction, reveal = true) {
    this.travelFrame++;
    let quicksandMultiplier = !reveal
      ? gs.agent.animations.simulation_multiplier
      : tile.isQuicksand
      ? gs.agent.animations.quicksand_multiplier
      : 1;
    const totalFrames = gs.agent.animations.travel_frames * quicksandMultiplier;

    // may want to multiply by direction distance to handle stuff like diagonal movement
    const t = this.travelFrame / totalFrames;
    if (this.travelFrame < totalFrames) {
      this.setPosition(tile.x + direction[0] * t, tile.y + direction[1] * t);
      return false;
    } else {
      this.setPosition(tile.x + direction[0], tile.y + direction[1]);
      this.travelFrame = 0;
      return true;
    }
  }
}
