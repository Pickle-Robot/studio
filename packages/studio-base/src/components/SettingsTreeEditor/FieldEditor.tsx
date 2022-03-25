// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import {
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  styled as muiStyled,
  MenuItem,
  Select,
  TextField,
} from "@mui/material";
import { Fragment } from "react";

import { SettingsTreeField } from "./SettingsTree";
import { ColorPickerInput } from "./inputs/ColorPickerInput";
import { ColorScalePicker } from "./inputs/ColorScalePicker";
import { NumberInput } from "./inputs/NumberInput";

const StyledToggleButtonGroup = muiStyled(ToggleButtonGroup)(({ theme }) => ({
  background: theme.palette.grey[200],
  padding: theme.spacing(0.25),
  border: `1px solid ${theme.palette.divider} !important`,
  gap: theme.spacing(0.25),

  ".MuiToggleButton-root": {
    borderRadius: `${theme.shape.borderRadius} !important`,

    "&.Mui-selected": {
      border: `1px solid ${theme.palette.divider} !important`,
    },
  },
}));

const StyledToggleButton = muiStyled(ToggleButton)(({ theme }) => ({
  borderRadius: `${theme.shape.borderRadius} !important`,
  padding: 0,
  border: "none",

  "&.Mui-selected": {
    background: theme.palette.background.paper,
    border: `1px solid ${theme.palette.divider}`,
  },
}));

export function FieldEditor({
  field,
  update,
}: {
  field: SettingsTreeField;
  update: (value: unknown) => void;
}): JSX.Element {
  return (
    <Fragment>
      <Typography variant="subtitle2" color="text.secondary" noWrap title={field.label}>
        {field.label}
      </Typography>
      {/* <IconButton edge="end" size="small">
            <DataObjectIcon fontSize="small" />
          </IconButton> */}
      <div>
        {field.input === "number" && (
          <NumberInput
            size="small"
            variant="filled"
            value={field.value ?? 0}
            placeholder={field.placeholder}
            fullWidth
            onChange={(event) => update(Number(event.target.value))}
          />
        )}
        {field.input === "toggle" && (
          <StyledToggleButtonGroup
            fullWidth
            value={field.value}
            exclusive
            size="small"
            onChange={(_event, value) => update(value)}
          >
            {field.options.map((opt) => (
              <StyledToggleButton key={opt} value={opt}>
                {opt}
              </StyledToggleButton>
            ))}
          </StyledToggleButtonGroup>
        )}
        {field.input === "select" && (
          <Select
            size="small"
            fullWidth
            variant="filled"
            value={field.value}
            onChange={(event) => update(event.target.value)}
          >
            {field.options.map((opt) => (
              <MenuItem key={opt} value={opt}>
                {opt}
              </MenuItem>
            ))}
          </Select>
        )}
        {/* {prop.input === "messagePath" && <>Message path input</>} */}
        {field.input === "string" && (
          <TextField
            variant="filled"
            size="small"
            fullWidth
            value={field.value}
            onChange={(event) => update(event.target.value)}
          />
        )}
        {field.input === "boolean" && <>TODO: Boolean</>}
        {field.input === "gradient" && <ColorScalePicker color="inherit" size="small" />}
        {field.input === "color" && (
          <ColorPickerInput
            defaultValue={field.value?.toString()}
            value={field.value?.toString()}
            size="small"
            variant="filled"
            fullWidth
            onChange={(event) => update(event.target.value)}
          />
        )}
      </div>
    </Fragment>
  );
}
