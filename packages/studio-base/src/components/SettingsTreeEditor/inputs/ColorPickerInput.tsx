// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import { TextField, ButtonBase, styled as muiStyled, TextFieldProps } from "@mui/material";
import { useState } from "react";

import { fonts } from "@foxglove/studio-base/util/sharedStyleConstants";

const StyledTextField = muiStyled(TextField)({
  ".MuiInputBase-formControl.MuiInputBase-root": {
    padding: 0,
  },
  ".MuiInputBase-input": {
    fontFamily: fonts.MONOSPACE,
    alignItems: "center",
  },
});

const ColorSwatch = muiStyled("div", {
  shouldForwardProp: (prop) => prop !== "color",
})<{ color: string }>(({ theme, color }) => ({
  backgroundColor: color,
  height: theme.spacing(2),
  width: theme.spacing(2),
  margin: theme.spacing(0.625),
  borderRadius: 1,
  border: `1px solid ${theme.palette.getContrastText(color)}`,
}));

type ColorPickerInputProps = {
  defaultValue?: string;
  swatchOrientation?: "start" | "end";
} & TextFieldProps;

export function ColorPickerInput(props: ColorPickerInputProps): JSX.Element {
  const { defaultValue, swatchOrientation = "start" } = props;
  const [color, setColor] = useState<string>(defaultValue ?? "#000000");

  // TODOS:
  // - Add a color picker component
  // - Make its safe to type invalid strings into the field

  return (
    <StyledTextField
      {...props}
      defaultValue={defaultValue}
      onChange={(event) => setColor(event.target.value)}
      InputProps={{
        startAdornment: swatchOrientation === "start" && (
          <ButtonBase>
            <ColorSwatch color={color} />
          </ButtonBase>
        ),
        endAdornment: swatchOrientation === "end" && (
          <ButtonBase>
            <ColorSwatch color={color} />
          </ButtonBase>
        ),
      }}
    />
  );
}
