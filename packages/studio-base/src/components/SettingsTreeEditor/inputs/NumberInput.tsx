// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { IconButton, TextFieldProps, TextField, styled as muiStyled } from "@mui/material";
import { ChangeEvent, ReactNode, useState } from "react";
import { useKeyPress } from "react-use";

import { fonts } from "@foxglove/studio-base/util/sharedStyleConstants";

const StyledTextField = muiStyled(TextField)({
  ".MuiInputBase-formControl.MuiInputBase-root": {
    padding: 0,
  },
  ".MuiInputBase-input": {
    textAlign: "center",
    fontFamily: fonts.MONOSPACE,

    "&::-webkit-outer-spin-button, &::-webkit-inner-spin-button": {
      appearance: "none",
      margin: 0,
    },
  },
  "@media (pointer: fine)": {
    ".MuiIconButton-root": {
      visibility: "hidden",
    },
    "&:hover .MuiIconButton-root": {
      visibility: "visible",
    },
  },
});

const StyledIconButton = muiStyled(IconButton)({
  "&.MuiIconButton-edgeStart": {
    margin: 0,
    borderTopRightRadius: 0,
    borderBottomRightRadius: 0,
  },
  "&.MuiIconButton-edgeEnd": {
    margin: 0,
    borderTopLeftRadius: 0,
    borderBottomLeftRadius: 0,
  },
});

export function NumberInput(
  props: {
    increment?: number;
    iconUp?: ReactNode;
    iconDown?: ReactNode;
    value: number;
  } & TextFieldProps,
): JSX.Element {
  const { value, iconDown, iconUp, increment = 1 } = props;
  const [_value, setValue] = useState<number>(value);

  const [shiftPressed] = useKeyPress("Shift");

  const incrementAmount = shiftPressed ? increment * 10 : increment;

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    setValue(Number(event.target.value));
  };

  return (
    <StyledTextField
      {...props}
      value={_value}
      onChange={handleChange}
      type="number"
      InputProps={{
        startAdornment: (
          <StyledIconButton
            size="small"
            edge="start"
            onClick={() => setValue(_value - incrementAmount)}
          >
            {iconDown ?? <ChevronLeftIcon fontSize="small" />}
          </StyledIconButton>
        ),
        endAdornment: (
          <StyledIconButton
            size="small"
            edge="end"
            onClick={() => setValue(_value + incrementAmount)}
          >
            {iconUp ?? <ChevronRightIcon fontSize="small" />}
          </StyledIconButton>
        ),
      }}
    />
  );
}
